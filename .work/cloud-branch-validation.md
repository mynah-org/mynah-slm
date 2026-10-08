# V1 — per-commit validation matrix for the cloud branch

Status: **IN PROGRESS** (opened 2026-10-08; updated with every commit on `claude/relaxed-mendel-1un91i`)

Item: `PLAN.md` §0 V1.

## Task

This branch was written on a cloud VM with **no model weights, no GPU, one
x86 CPU** (4-vCPU Cascade Lake-class: AVX-512F/BW/VL + VNNI, no AMX, no
AVX512-BF16), plus qemu-aarch64 for NEON correctness. It is a set of
candidates to keep, fix or revert **one commit at a time** on real hardware:

```
git log --oneline main..claude/relaxed-mendel-1un91i
```

Each row says what was proven here, what was assumed, what is expected, and
exactly what to run downstream. A commit whose downstream check fails is
reverted on its own; the rows are ordered so that later commits depend only
on the ones they name.

## Legend

- **Here**: what ran in the cloud VM. `x86` = native AVX-512 / AVX2 / scalar
  builds; `qemu` = aarch64 NEON under qemu-user (correctness only, never
  speed); `TSan` = ThreadSanitizer; `-Werror` = `make warnings` with gcc 13
  and clang.
- **Downstream**: what only the user's machines can answer. "M1" = Apple
  M-series dev Mac, "x86" = the Linux x86 host(s), "Axion" = Neoverse V2,
  "L4/L40S" = NVIDIA. Weights LOCAL for anything that measures.

## Matrix

| Commit | Change | Here | Assumptions | Expected benefit | Downstream (exact) | Depends on |
|---|---|---|---|---|---|---|
| `2714c40` | K1 own f32 GEMM, opt-in `BLAS=none` | test_sgemm PASS x86 avx512/avx2/scalar + qemu NEON; TSan clean; `make test BLAS=none`; -Werror | default build unchanged (vendor still linked) | build with zero vendor deps; deterministic GEMM | `make clean && make BLAS=none test test-parity` (M1, x86, Axion); `tests/test_sgemm bench 1` on x86 + Axion; e2e prefill 198 / 2275 tokens `BLAS=none` vs default. Gate in `no-blas.md` | — |
| `5f23d86` | K2 pool: atomic claim, bounded spin, affinity | test_threads PASS x86 (x5) + qemu; TSan clean; `bench/pool_ab` 4x on a synthetic 80 us region, half the CPU per burst | spin 50 us suits decode gaps | +5-15% decode on 0.6B (prediction) | decode tok/s old vs new vs `MYNAH_SLM_POOL_SPIN_US=0`, 0.6B Q4_K_M, M1 8t, x86, Axion; `make test-server`; `bench/pool_ab/run.sh` | — |
| `70bbe7d` | CI rows `BLAS=none` x86_64 + aarch64 | YAML parses; `make test BLAS=none` locally | hosted runners as today | keeps the dependency-free build green | watch the two new rows on the PR | `2714c40` |
| `923576b` | notes + `bench/pool_ab` | check_plan OK | — | — | `bench/pool_ab/run.sh` on each host | `5f23d86` |
| `0860c56` | board + K2 note + this matrix | check_plan OK | — | — | — | — |
| `7e904f6` | pool: nested / concurrent parallel_for run inline (review finding) | test_threads nested + two-caller cases PASS native, TSan, qemu | one region at a time is the contract | no hang if the scheduler or a future caller nests | `make test` on M1 / Axion (real weak-memory ARM: qemu-user is TSO and proves nothing about ordering); `/tmp`-style stress from the review: 2 caller threads x 20000 regions | `5f23d86` |
| `1b29990` | sgemm: explicit store rounding, AVX-512 needs DQ, no software fmaf (review findings) | test_sgemm PASS gcc+clang x c11/gnu11 x -ffp-contract=fast x avx512/avx2/scalar/avx512f-no-dq, qemu NEON; mutation (old store) FAILS the new beta=0.7 case | — | per-build bit identity under any contraction mode; scalar 0.8 -> 5-7 GFLOP/s | `make BLAS=none test` with Apple clang on M1 (its default contraction differs) | `2714c40` |
| `e39e0bd` | G1 note: CUDA design, Qwen3 kernel list, integration plan | check_plan OK | mynah-tts host-overhead lessons transfer | plan before code | review | — |
| `acaeecf` | backend vtable + CPU backend over existing kernels (`src/backend*.{c,h}`, `tests/test_backend.c`) | every op memcmp-identical to the direct call (F32/Q8_0/Q4_K/Q6_K matvec/matmat, RoPE, norms, attention f32/bf16/q8 KV); ASan+UBSan; -Wpedantic -Werror gcc/clang; BLAS=none | qmat.h/kernels.h signatures stable | a seam for GPU without touching the CPU reference | `make clean && make test` on M1, x86, Axion (NEON) | — |
| `b675eec` | CUDA backend behind `make cuda` (`gpu/cuda/`): F32/Q8_0/Q4_K/Q6_K GEMV, RMSNorm, QK-norm, NeoX RoPE, SwiGLU, bf16 KV append, GQA decode attention (head_dim 64/128/256), argmax | compiles+links sm_80/89/90 with nvcc 12.0 (-Werror); host-side decode/f16/bf16/RoPE helpers bitwise vs ingot/src; no device -> exit 77 cleanly. **NO KERNEL HAS RUN** | launch geometry, warp reductions, online softmax correct | correct-first device decode step | `make cuda-test CUDA_ARCH=sm_89` on L4/L40S (every line `ok`); `compute-sanitizer --tool memcheck build/cuda/test_cuda`; `--tool racecheck` | `acaeecf` |
| `861b973` | CI compile-only CUDA job (nvidia/cuda 12.6.2 container, sm_80/89/90) | YAML parses; same steps pass locally on 12.0 | container toolkit ~ 12.0 | CUDA build cannot rot | watch the 3 CUDA rows on the PR | `b675eec` |
| `e7af121` | slot pool with non-synchronizing release (cancelled requests never cudaFree), backend recover | CPU test: capacity, generation bump, double release refused, reuse keeps pointers, reused KV == fresh KV bitwise; ASan+UBSan; make cuda builds | serving drops results by generation and rebuilds per-row tables per step | a cancelled client costs at most one in-flight GPU step | `make cuda-test` (`check_slots`); nsys during a cancel burst: no cudaFree / cudaStreamSynchronize on retire | `b675eec` |
| `0ec76cd..ea0477b` (9 commits, see `git log`) | CUDA review fixes: uploads ordered on the backend stream (B1, a real race), device selected per op (R1), drain before frees + `weights_flush` + lifetimes (R3-R5), host half built `-ffp-contract=off` (N2), NaN-safe argmax matching the CPU (N1), self-test covers pos 0-2 / batch from 0 / 1026 rows with exact gates where exact (R2/N3/N4), slot-reuse isolation test that can fail (N5), CI hardening (N7) | each commit: make test, -Werror gcc+clang, check_plan, `make cuda` sm_89 BLAS=none and sm_80, test_cuda exit 77 no FAIL; argmax and N5 mutation checks fail as intended; reviewer's warp emulator 0 failures | the CPU reference is NOT FMA-fused (true for gcc -std=c11; clang -march=native on M1 may fuse, which would make the exact decode/RoPE device gates fail for no device bug) | the CUDA foundation can be run for the first time without known host-side races | on L4/L40S: `make cuda-test CUDA_ARCH=sm_89`, then `compute-sanitizer --tool memcheck build/cuda/test_cuda` and `--tool racecheck`; on a 2-GPU host `MYNAH_SLM_CUDA_DEVICE=1 build/cuda/test_cuda` | `e7af121` |
| `155ad48` | K3: opt-in int8 Q4_K kernel, 4 rows per activation load, one reduction per row (AVX-512 VNNI, AVX2, NEON SDOT) + scalar twin; `tests/bench_qmat`; fixes qmat.c not compiling for armv8-a | memcmp vs twin on avx512vnni (gcc+clang), avx2, qemu NEON; tails, 1-vs-4 threads; twin vs ingot dequant 1.2-2.2e-7; mutation caught; interleaved old vs new 2.2-2.4x (AVX-512), 1.9-2.1x (AVX2) on layer shapes, 15/15 wins, head 1.25x/tie | the 1.4-3.9e-7 reorder does not move perplexity | `--fast` decode +5-15% on x86 (prediction); default path untouched | M1, x86, Axion: `make clean && make test test-parity`; `tests/bench_qmat k3 21 1,<n>`; decode `--fast` vs not, interleaved; `MYNAH_SLM_INT8=1 MYNAH_SLM_INT8_TYPES=q4_k ./mynah-slm ppl ...` (expect 2.842 +-0.002) | — |
| `ae495b5` | K4: int8 Q8_0 and Q6_K matvecs under `--fast`; `MYNAH_SLM_INT8_TYPES` narrows per type | byte-exact vs twins all ISAs + qemu; mutations caught; without int8 still ingot (asserted); vs ingot f32, AVX-512 1t: Q6_K 4.3-5.5x layers, 1.63-1.66x on the 151936x1024 head; Q8_0 2.7-3.1x; AVX2 2.2-2.6x; two sessions agree | int8 on the Q6_K head (in front of the softmax) is acceptable for quality | further `--fast` decode on Q4_K_M from the head, +15-30% x86 (prediction) | `tests/bench_qmat k4 21 1,<n>`; ppl with `MYNAH_SLM_INT8_TYPES=q4_k` vs `q4_k,q6_k`, and on a Q8_0 file; decode both sets; if M1 Q6_K < 1.05x narrow on ARM | `155ad48` |
| `2d9382a` | K6 REJECTED (note + `bench_qmat k6` roof), no kernel | ingot bf16/f16 matvec at 0.87-0.94x of a pure read of the same bytes; no shipped 2-byte format | 2-byte decode is DRAM-bound on targets too | none; avoids untimed kernels | Neoverse V2, Zen4/SPR, M2+: `tests/bench_qmat k6 21 1,<n>`; reopen only if < 0.8x AND a BF16 rung ships | — |
| `a3e10fe` | K5: per-ISA kernel TUs (`src/kern.h`, `qmat_kern.c`, `attn_kern.c`), CPUID/getauxval/sysctl probe (`src/isa.c`), narrow-only `MYNAH_SLM_ISA`, verify-on-first-use (`src/isa_verify.c`), `mynah-slm --dispatch`, `tests/test_isa.c` | x86-64-v2 binary resolves qmat avx512vnni / attn avx2 like native; stage hashes identical pre-K5 / native / v2; aarch64 armv8-a under qemu: neon_dotprod on -cpu max, neon on cortex-a53; verify rejects a 1-ulp deviation; dispatch overhead ~1.01x; TSan/UBSan/-Werror | probes behave on real silicon (M1 sysctl, Neoverse getauxval, Zen4/Skylake-SP feature mixes) | portable tarballs reach the fast kernels | every host: `./mynah-slm --dispatch`; portable build (`ARCH_FLAGS="-march=x86-64-v2 -mtune=generic"` / `-march=armv8-a`) then `--dispatch`; `MYNAH_SLM_ISA=avx2|scalar`; decode portable vs native interleaved; `make test test-parity`. NOTE: ingot's own kernels are still compile-time, so a portable build runs ingot's Q6_K/Q8_0 scalar without `--fast` (flagged by --dispatch; fix belongs upstream) | `ae495b5` |
| `ff349ac` | K5 for sgemm: sgemm.c also compiled as kernel TUs (x86 scalar/avx2/avx512, arm64 scalar/neon); integration kept the review fixes (SG_FMA1, AVX512DQ guard) | v2 resolves sgemm avx512; hashes identical; own vs ref < 1e-5 and 4t == 1t at every level; test_sgemm, BLAS=none test, qemu both CPUs | — | only with BLAS=none / MYNAH_SLM_SGEMM=own | `make BLAS=none test`; `tests/test_sgemm bench 1` on portable builds | `a3e10fe` |
| `1521ceb` | integration: link the kernel TUs into `make cuda` | `make cuda` sm_89 and sm_80 BLAS=none link; test_cuda exit 77; `--dispatch` OK | — | — | `make cuda-test` on L4/L40S | `a3e10fe`, CUDA commits |
