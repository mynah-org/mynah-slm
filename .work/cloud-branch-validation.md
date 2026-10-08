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
