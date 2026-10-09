# K1 — our own f32 GEMM, and the gate before it becomes the Linux default

Status: **IN PROGRESS** (landed opt-in 2026-10-08 as `BLAS=none`; default flip OPEN)

Item: `PLAN.md` §0 K1. Lineage: mynah-tts `src/sgemm.c` and its
`.work/no-blas.md` ("OpenBLAS leaves the process for good"). Map:
[`sibling-port-map.md`](sibling-port-map.md) row 2.
Code: `src/sgemm.{c,h}`, `tests/test_sgemm.c`, `Makefile` (`BLAS=`).

## Task

Take the vendor BLAS out of the process without making prefill slower.

## Question

Two separate questions, and only the first is answered:

1. Can the three GEMM call sites run on our own code, deterministically, on
   every ISA we build for? **Yes — proven locally.**
2. Is it fast enough to be the default? **No — measured 0.3-0.85x OpenBLAS on
   the shapes that dominate prefill, on the one machine available.**

## Known facts

- Three call sites, all prefill or batched attention; decode never reaches a
  GEMM (`src/qmat.c` qmatmat, `src/kernels.c` attention_batch and
  attention_kv_batch). All now call `mynah_slm_sgemm`, which forwards to the
  vendor when one is linked — **the default build's numerics are unchanged.**
- Why mynah-tts moved and why the reason holds here: a second thread pool in
  the address space (OpenBLAS's) made profiles environment-dependent, and
  mynah-asr measured concurrent inferences on one OpenBLAS pool collapsing
  aggregate throughput. S1 (batched serving) makes that our problem too.
- Why their kernel does NOT transfer as-is: their dominant shape is
  m=512 n=16 k=512 (a 16-frame batch). Ours is out[T][128] = in[T][K] · W[128][K]^T
  with T up to 256 — an NT product — plus QK^T at k = head_dim = 128. Their
  NT family is "one dot per element"; that is the shape we needed most.

## What landed

- **NT family**: an MR×NR register tile of dot products, k vectorised, no
  packing; rows of A processed in L2-sized chunks (256 KiB) so A is not
  re-streamed from L3 once per column tile.
- **NN family**: the mynah-tts broadcast micro-kernel, k-blocked (256) when
  beta == 0 by parking the raw f32 accumulator in C — exact.
- **Determinism stronger than mynah-tts's**: every element is computed by the
  same operation sequence whatever tile, chunk, k-block or thread holds it, so
  results are bit-identical across thread counts AND tilings. Asserted by
  memcmp in `tests/test_sgemm.c`.
- `BLAS=auto|none|openblas|accelerate`; auto keeps OpenBLAS (Linux) /
  Accelerate (macOS). `MYNAH_SLM_SGEMM=own` routes a vendor build to ours at
  run time for an in-process A/B.
- CI rows that build and test `BLAS=none` on x86_64 and aarch64 **without
  OpenBLAS installed**.

## Evidence

### Proven locally (no model, no special hardware)

| Check | x86 AVX-512 | x86 AVX2+FMA | x86 scalar | aarch64 NEON |
|---|---|---|---|---|
| `tests/test_sgemm` (reference agreement < 1e-5 rel, 1-vs-4-thread memcmp, tile/chunk/k-block invariance memcmp, NaN-poisoned C with beta=0, ldc padding) | PASS native | PASS native | PASS native | PASS under qemu-aarch64 (correctness only; qemu says nothing about speed) |
| Same test under ThreadSanitizer (x86) | clean | — | — | — |
| `make test` with `BLAS=none` (model tests SKIP) | PASS | — | — | — |
| `make warnings` (-O2 -Werror), gcc 13 and clang | PASS | — | — | — |

### Measured locally — performance (CLOUD-SPECIFIC, revalidate on target)

Machine: cloud VM, 4 vCPU "Intel Xeon Processor @ 2.80GHz", family 6 model 85
(Cascade Lake class: AVX-512F/BW/VL + VNNI, no AMX), 1 MiB L2 per core, 33 MiB
L3, kernel 6.18, gcc 13.3 `-O3 -march=native` (our ISA resolved: avx512).
Vendor: OpenBLAS 0.3.26 (Ubuntu `openblas-pthread`). f32. Shapes are what
Qwen3-0.6B issues at the default prefill width 256 and at short prompt tails.
Method: `tests/test_sgemm bench <thr>` — 15 rounds, own and vendor timed back
to back with the order alternating per round, ~0.3 GFLOP per timed block.
"Direction stable" = one side won ≥ 14/15 rounds; "magnitude stable" = both
coefficients of variation < 5%. Two independent sessions at 1 thread
(same code; session A = after the blocking change, session B = final HEAD).

**1 thread** (`OPENBLAS_NUM_THREADS=1`) — the clean comparison:

| Shape | M | N | K | op(B) | own GF/s (B) | OpenBLAS GF/s (B) | ratio A | ratio B | own wins (B) | direction | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qmatmat q_proj T=256 | 256 | 128 | 1024 | B^T | 42.3 | 73.1 | 0.58x | 0.66x | 0/15 | stable: **slower** | no (CV 12% / 24%) |
| qmatmat ffn_down T=256 | 256 | 128 | 3072 | B^T | 37.6 | 45.4 | 0.65x | 0.83x | 1/15 | stable: **slower** | no (CV 14% / 26%) |
| qmatmat ffn_up T=64 | 64 | 128 | 1024 | B^T | 69.7 | 66.1 | 1.05x | 1.06x | 14/15 | stable: faster | yes |
| qmatmat ffn_up T=16 | 16 | 128 | 1024 | B^T | 65.8 | 32.6 | 2.05x | 2.03x | 15/15 | stable: **faster** | yes |
| qmatmat ffn_up T=7 | 7 | 128 | 1024 | B^T | 51.1 | 60.1 | 0.82x | 0.85x | 0/15 | stable: slower | yes |
| QK^T n_q=256 n_kv=256 | 256 | 256 | 128 | B^T | 26.8 | 90.7 | 0.29x | 0.30x | 0/15 | stable: **much slower** | yes |
| QK^T n_q=256 n_kv=2048 | 256 | 2048 | 128 | B^T | 22.1 | 73.1 | 0.31x | 0.30x | 0/15 | stable: **much slower** | borderline (5.0%) |
| SV n_q=256 n_kv=256 | 256 | 128 | 256 | B | 46.5 | 91.1 | 0.52x | 0.51x | 0/15 | stable: slower | yes |
| SV n_q=256 n_kv=2048 | 256 | 128 | 2048 | B | 32.7 | 75.9 | 0.62x | 0.46x | 0/15 | stable: slower | no (CV 17% / 24%) |

**4 threads** (`OPENBLAS_NUM_THREADS=4` vs our pool at 4) — NOT a clean
comparison, kept for the record only. Every row has CV 10-45% on at least one
side, and interleaving two thread pools on 4 vCPUs contaminates both: whichever
pool ran last is still spinning (ours for up to 50 us, OpenBLAS for its own
idle spin) while the other one is timed. Ratios ranged 0.26x (QK^T n_kv=2048)
to 1.35x (T=7). No conclusion is drawn from this half.

What the 1-thread table supports, and nothing more:

- **On this VM**, ours wins only on short batches (T ≤ 64, i.e. prompt tails)
  and loses 0.3-0.85x everywhere prefill spends its time.
- The QK^T gap (0.3x) is structural, not noise: at k = 128 the NT tile does 8
  vector FMAs per element on AVX-512 and then a horizontal sum that costs about
  as much. A dot-product formulation is the wrong one for short k.
- Earlier, a one-shot run (one round, no alternation) showed own at 1.12-1.41x
  on T=64/T=33. Repeated, T=64 holds at ~1.05x and the large shapes do not.
  Recorded because it is the exact trap the method warns about.

## Unknowns — downstream validation (needs the user's hardware / a model)

1. End-to-end prefill tok/s and TTFT with `BLAS=none` vs default on
   Qwen3-0.6B Q4_K_M, 198 and 2275-token prompts (the docs/perf.md pair),
   weights LOCAL, same threads. GEMM is only part of prefill (the strip
   dequant runs on the pool either way), so the e2e gap should be smaller
   than the kernel gap — **prediction, not measured**: 0.75-0.95x prefill on
   x86 at the current kernel.
2. `make test-parity` and `tests/test_batch` with `BLAS=none`: the batch-vs-
   decode reorder gate (1.5e-6 measured with Accelerate) must still hold.
3. Apple M-series and Neoverse V2: none of the numbers above transfer.

## Gate for making `BLAS=none` the Linux default

All of, on the target Linux hosts (x86 and ARM), weights local:

- `make test-parity` and `test_batch` pass with `BLAS=none`;
- `tests/test_sgemm bench 1`: ratio ≥ 0.9x on every T=256 and attention row,
  direction stable;
- e2e prefill tok/s ≥ 0.95x the OpenBLAS build at 198 and 2275 tokens.

## Next action

The kernel work that would close the gap, as a separate item with its own A/B
— not in this PR:

- **NT at large T**: pack the 128-row weight strip once per qmatmat into
  [K][NR] panels and run a broadcast micro-kernel (2-D register blocking, e.g.
  AVX-512 8×32, NEON 8×12). The strip is already being written by our own
  dequant; writing it pre-packed may make the packing free.
- **QK^T at k = 128**: switch to NN by gathering K^T for the KV head once (the
  kv_batch path already gathers K into scratch — gather it transposed).
