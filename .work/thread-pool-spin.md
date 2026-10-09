# K2 — the pool: atomic claiming, bounded spin, affinity width

Status: **IN PROGRESS** (landed 2026-10-08 in `src/threads.c`; decode tok/s UNMEASURED)

Item: `PLAN.md` §0 K2. Lineage: mynah-tts `src/threads.c` (spin calibration
:145-290, atomic claiming :694-705, affinity sizing :55-90) and its
`.work/pool-barrier-meter.md`. Map: [`sibling-port-map.md`](sibling-port-map.md) row 1.
Code: `src/threads.{c,h}`, `tests/test_threads.c`, `bench/pool_ab/`.

## Task

Cut the per-region cost of `mynah_slm_parallel_for`, which a Qwen3-0.6B decode
step calls ~225 times.

## Question

Is region dispatch a material share of a decode step, and does the mynah-tts
pool design remove it without changing a single output bit?

## Known facts

- Old pool: one global mutex taken twice per task, workers park on a condvar
  after every region, width from `sysconf` (whole machine, ignoring taskset).
- mynah-tts measured a cold condvar wake at 22-29 us; a region needs ~200 us
  of work to pay for its dispatch. Our decode regions average ~120 us on M1
  (36.5 tok/s, 8 threads, ~225 regions) — inside the range where dispatch
  dominates.
- Results stay bit-identical by construction: tasks write disjoint slices and
  the caller waits on tasks finished, not workers present (this repo's own
  determinism bug, documented in `threads.c`).

## What landed

- Claim = one CAS on a word packing generation (24 bits), task count (20) and
  next index (20). **The count must share the word with the index**: the first
  version kept it in a separate field, a late worker read the NEXT region's
  larger n against a still-exhausted claim, won the CAS and ran a task that
  did not exist. `tests/test_threads.c` caught it as a hang (done = 96 for a
  95-task region). Regions wider than 2^20 tasks are split, never truncated.
- Workers and the waiting caller spin for a bounded time
  (`MYNAH_SLM_POOL_SPIN_US`, default 50 us, `0` = old park-at-once
  behaviour), then park. Publish-then-lock / park-under-lock protocol, so the
  spin is never needed for correctness.
- Width from `sched_getaffinity` on Linux.

## Evidence

### Proven locally

| Check | Result |
|---|---|
| `tests/test_threads`: 65k regions of random width (1-256 tasks), spin 0 / 5 / 50 us, 1-8 threads; every task exactly once; never run under another region's function; repeated resize | PASS x86 native (5 consecutive runs), PASS qemu-aarch64 |
| Same under ThreadSanitizer | clean |
| `tests/test_sgemm` (uses the pool at 4 threads, memcmp vs 1 thread) | PASS |

### Measured locally — CLOUD-SPECIFIC, revalidate on target

Machine: 4-vCPU Cascade Lake-class VM (see [`no-blas.md`](no-blas.md)), gcc 13
-O2. `bench/pool_ab/run.sh 4`: old pool (from `cde1599`, untouched) vs current,
same source, alternating processes, 10 rounds. Regions are nth*4 = 16 tasks of
synthetic multiply-adds; "serial" is the same work with no pool.

| region | serial us | old us (median, min..max, CV) | new us | new, spin=0 us | old/new | new wins |
|---|---|---|---|---|---|---|
| empty | - | 12.0 (5.3..22.7, 41%) | 3.3 (2.0..4.0, 20%) | 5.0 (2.0..8.8, 44%) | 3.60x | 10/10 |
| tiny (256 MAC/task) | 4.8 | 34.8 (20.5..42.1, 19%) | 3.5 (2.4..4.1, 15%) | 24.0 (16.6..38.2, 26%) | 9.92x | 10/10 |
| small (4 K MAC/task) | 82.5 | **100.8** (87.9..113.0, 6%) | **25.2** (21.9..30.5, 11%) | 100.4 (89.6..117.3, 8%) | 4.00x | 10/10 |
| medium (32 K MAC/task) | 663.9 | 283.7 (264.5..289.8, 3%) | 200.4 (185.4..221.0, 6%) | 273.3 (254.7..301.8, 4%) | 1.42x | 10/10 |

| burst: 225 "small" regions + 1 ms idle gap, x100 | old | new |
|---|---|---|
| wall ms per step | 25.0 (22.2..27.5, CV 6%) | 7.6 (6.9..8.3, CV 6%) |
| CPU ms per step (all threads) | 43.4 (39.1..46.9, CV 5%) | 22.7 (20.6..23.6, CV 5%) |

What it supports:

- **The old pool made a ~80 us region slower than not threading at all**
  (100.8 vs 82.5 us). The new one is 3.3x faster than serial on 4 cores.
- **The win is the spin, not the CAS**: spin=0 (atomic claim, park at once)
  lands back on the old numbers in every row.
- **It costs less CPU, not more**: half the CPU time per burst. The spin is
  bounded to 50 us; the old pool spent more than that in futex traffic and
  mutex contention. This matters for running beside the ASR and TTS stages.
- Direction stable in every row (10/10); magnitudes are VM-noisy (CV up to 44%
  on the empty row).

## Unknowns — downstream validation

1. **Decode tok/s**, Qwen3-0.6B Q4_K_M, weights LOCAL, same threads, old vs
   new binary — and `MYNAH_SLM_POOL_SPIN_US=0` on the new binary as the
   in-binary control. **Prediction, written before the run**: +5-15% decode on
   0.6B at 8 threads (225 regions x ~10-20 us saved of ~27 ms), shrinking
   toward 0 on 4B/8B where regions are long and bandwidth-bound.
2. Apple M-series: P/E cores. A spinning E-core worker may hold a region
   open; counter-based claiming should absorb it, but measure 4 vs 8 threads
   again.
3. `bench/pool_ab/run.sh` on the target hosts (x86 and Neoverse), plus
   `make test-server` (the five-identical-answers determinism check) with the
   new pool.
4. CPU budget beside ASR+TTS: `bench/pool_ab/burst_cpu_*` with a realistic gap
   (sampling + TTS hand-off), and the spin budget swept 0/10/25/50/100 us.

## Next action

Downstream validation above. If decode does not move on 0.6B, the region
count is the next lever (fuse q/k/v into one region, gate+up into one), not a
larger spin.
