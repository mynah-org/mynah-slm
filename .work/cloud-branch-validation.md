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
