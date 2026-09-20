# R1 — claim audit

Status: **IN PROGRESS** (opened 2026-09-20)

Item: `PLAN.md` §0 R1-A. Audits every load-bearing conclusion currently in
[`ternary-feasibility.md`](ternary-feasibility.md) and
[`ptqtp-paper-reading.md`](ptqtp-paper-reading.md).

**This note exists to falsify our own conclusions, not to defend them.** A claim
that survives is worth more afterwards; a claim that does not was costing us.

## Evidence labels — mandatory from here on

| label | meaning |
|---|---|
| `[MEASURED]` | we ran it, on stated hardware, with a recorded protocol |
| `[PAPER]` | a published claim by someone else, cited to section/table |
| `[ARTIFACT]` | read off a released checkpoint's bytes |
| `[DERIVED]` | arithmetic on other labelled facts; no new observation |
| `[HYPOTHESIS]` | a proposed explanation with no decisive evidence yet |
| `[UNKNOWN]` | we cannot currently tell, and know why |

A `[HYPOTHESIS]` is never silently promoted. Where evidence conflicts, **the
conflict is preserved in the note** rather than resolved by preference.

## New primary source read for this audit

**Tied Trit-Planes** (arXiv 2608.08910, Grella, Aug 2026) — PDF fetched and
extracted locally. It matters here because it is the only source that has
actually *built and measured* CPU SIMD kernels for a PTQTP-derived
representation, on both aarch64 and x86-64. Its subject is a 284B MoE with SSD
expert streaming, **so nothing in it transfers to dense Qwen3 by assumption** —
but it is direct counter-evidence to some of our claims about what is possible.

---

## C1 — "Qwen3-0.6B decode is ALU-bound"

**EVIDENCE**
- `[MEASURED]` Thread sweep, clean tree, local weights, deterministic workload:
  12.8 / 22.2 / 37.8 / 37.1 tok/s at 1/2/4/8 threads — **2.95× on four cores**,
  peak **14.8 GB/s** of weight traffic.
- `[MEASURED]` `--fast` (int8 activations, **zero** weight-byte change) buys
  **+48%** at one thread.
- `[MEASURED]` `docs/perf.md`: on granite-350m, Q8_0 decodes 54.8 tok/s against
  Q4_K_M's 42.2 **while reading 1.6× the bytes**.
- `[MEASURED]`, sibling repo: the PocketTTS backbone sustained **32–38 GB/s** on
  this machine class and scaled only **1.15×** on eight cores.

**CLASS** `[MEASURED]` for the stated configuration.

**COUNTER-EVIDENCE**
- The Q8_0-beats-Q4_K_M result is on **granite-350m**, not Qwen3-0.6B. Different
  model, different tensor mix. `[MEASURED elsewhere]`
- The 32–38 GB/s reference comes from a **different binary and a different
  model**. Using it as this machine's bandwidth ceiling is `[DERIVED]`, not
  measured on our engine.
- The 4→8 thread plateau is *also* consistent with hitting a bandwidth ceiling
  at four threads. 14.8 GB/s makes that implausible for an M1, but we have not
  measured this binary's achievable bandwidth directly.

**WHAT WOULD FALSIFY IT** A STREAM-class measurement of our own binary showing
it cannot exceed ~15 GB/s; or a format that halves weight bytes at equal kernel
complexity and delivers a proportional decode speedup.

**STATUS** **SUPPORTED, narrowly.** True of *Qwen3-0.6B at Q4_K_M on this M1
with our current kernels*. **Not** established as a property of the model, of
the architecture, or of any other ISA.

---

## C2 — "Therefore ternary cannot materially improve CPU decode"

**EVIDENCE** An inference from C1 plus the Phase 7a traffic model. No ternary
kernel was ever benchmarked, by us or by anyone we had read at the time.

**CLASS** `[HYPOTHESIS]` — it was written as a conclusion. That was the error.

**COUNTER-EVIDENCE**
- `[PAPER]` Tied Trit-Planes §4.4 **measures** a PTQTP-derived nine-level format
  at 4.0625 bpw decoding **6.7% faster** than `q4_k`, with real NEON `sdot` and
  AVX2/AVX-VNNI `vpdpbusd` kernels: 30.31 GB read per 32-token round against
  `q4_k`'s 33.89 GB (−10.6%), 10.7–10.9 s against 11.8–12.0 s of blocked-read
  time.
- `[DERIVED]` That win **tracks the bits/weight ratio**, so it is a traffic win
  in a disk-streamed regime, not an arithmetic one — which is consistent with
  C10's arithmetic half, and inconsistent with C2 as stated.

**WHAT WOULD FALSIFY IT** Exactly what the Tied paper did: build the kernel and
measure. In the opposite direction, a matched microbenchmark on a dense model
where a ternary GEMV loses to Q4 at equal thread count.

**STATUS** **NOT ESTABLISHED. Withdrawn.** The regime matters: their win is in a
bandwidth-dominated MoE streaming setting, ours is a dense model we measured as
ALU-bound on one machine. Neither settles the other.

---

## C3 — "Below ~3 bits nothing survives on Qwen3-0.6B"

**EVIDENCE** `[MEASURED]`, one llama.cpp harness, 60 identical chunks, imatrix
from WikiText-2 train: `IQ2_XXS` (2.06 bpw) **101×** baseline; `IQ1_S` (1.56 bpw)
**362×**; our naive single-plane W1.58 (1.69 bpw) **32,648×** on the HF harness.

**CLASS** `[MEASURED]` for *the methods tested*. The universal form is
`[HYPOTHESIS]`.

**COUNTER-EVIDENCE**
- No genuine sub-3-bit *survivor* has been tested, so the claim is untested in
  the only direction that could break it. `IQ3_XXS` at 3.06 bpw survives at
  1.86×, which places the cliff between 2.06 and 3.06 — but nothing was measured
  inside that interval.
- `[PAPER]` The Tied paper's dissociation result (C11 below) means a perplexity
  collapse is not automatically a capability collapse. We have not run a single
  behavioural eval on any of these.

**WHAT WOULD FALSIFY IT** Any sub-3-bit representation scoring under ~2× on this
model — `IQ2_M`, `IQ2_S`, `TQ2_0` from a real ternary fit, or a mixed layout at a
sub-3-bit average.

**STATUS** **RESTATE.** The defensible claim is: *"every conventional sub-3-bit
method we tested degraded Qwen3-0.6B by two orders of magnitude, and the cliff
lies between 2.06 and 3.06 bpw."* The universal form is retracted.

---

## C4 — "The collapse is a property of the small 0.6B model"

**EVIDENCE** `[PAPER]` PTQTP Table 1, ppl ratios computed from their own numbers:

| model | FP16 | PTQTP | ratio |
|---|---|---|---|
| Qwen3-0.6B | 20.9 | 38.02 | **1.819** |
| Qwen3-1.7B | 16.70 | 32.46 | **1.944** |
| Qwen3-4B | 13.64 | 18.25 | 1.338 |
| Qwen3-8B | 9.71 | 11.8 | 1.215 |
| Qwen3-32B | 8.64 | 10.06 | 1.164 |

**COUNTER-EVIDENCE, and it is ours** Earlier notes said *"the degradation is
monotone in model size and 0.6B is the worst case in the paper."*
**That is false on the paper's own table: 1.7B is worse than 0.6B** (1.944
against 1.819). The monotone reading was written while quoting numbers that
contradict it — an error of narrative over arithmetic, and exactly the class of
mistake this audit exists to catch.

**CLASS** `[HYPOTHESIS]`, and partially contradicted by the source it was drawn
from.

**WHAT WOULD FALSIFY IT** Our own harness on 1.7B and 4B. If 1.7B is worse than
0.6B on our measurements too, "small models are fragile" is the wrong model of
what is happening.

**STATUS** **NOT ESTABLISHED, and the supporting claim was misread.** Phase F is
the test. Correction propagated to `ternary-feasibility.md`.

---

## C5 — "PTQTP is effectively the same size as our Q4_K_M"

**EVIDENCE** `[DERIVED]` 369.4 MiB against 372.7 MiB — from 4.25 bpw on 59.1%
coverage with a Q6_K embedding.

**COUNTER-EVIDENCE**
- `[PAPER]` Tied Trit-Planes states the free-scale (untied) two-plane serving
  format is **4.125 bpw**, and its own tied fold is **4.0625 bpw**. Both are
  below our 4.25 assumption, which used PTQTP's G=128 rather than a 256-element
  block.
- `[DERIVED]` A base-3 packed form is 3.375 bpw. The size therefore spans
  **3.375 → 4.25** depending on a packing choice, and the claim silently fixed
  the least favourable end.

**STATUS** **SUPPORTED only for one configuration**, and the configuration was
not flagged as a choice. Restated in Phase H with all three numbers separated.

---

## C6 — "q_proj/k_proj were protected because they are sensitive"

**EVIDENCE** Split into what is actually known:

- `[ARTIFACT]` In all 28 blocks, `q_proj` and `k_proj` show 121–128 distinct
  values per 128-group where the other five families show exactly 9.
- `[ARTIFACT]`, and this is the decisive one:
  `‖W_artifact − W_orig·s‖/‖W_artifact‖ = 2.9e-04` for `q_proj` L13, where `s` is
  the per-channel scale recovered from the layernorms. **The artifact's q_proj is
  the original times a channel scale, with no quantization at all** — a ternarized
  tensor rescaled afterwards would leave a residual of ~0.18, not 2.9e-04. So the
  distinct-value count alone would have been ambiguous (a per-column rescale can
  turn 9 levels into 128 distinct values); the residual test is what settles it.
- `[MEASURED]` Our Method C is worse when q/k are included: 59.539 against
  43.331.
- `[UNKNOWN]` **Why** the released checkpoint treats q/k differently. The paper
  says the opposite (§4.1: "All linear layers were quantized") and gives no rule.

**STATUS** The *observation* is `[ARTIFACT]` and solid. The *motive* is
`[UNKNOWN]`. Our own note already hedged this; the audit adds the residual test
as the reason the observation is safe, and Phase C will measure whether q/k are
in fact disproportionately sensitive.

---

## C7 — "Scale absorption explains the gap between our Method C and the artifact"

**EVIDENCE** `[MEASURED]` Four conditions, none reproduce the artifact:

| form | α | result |
|---|---|---|
| weight-only | 0 → 1 | reconstruction flat at 0.179 (±0.1%) |
| activation-aware | +0.5 | ppl 62.32 (**+19** against no absorption) |
| activation-aware | −0.5 | ppl 781.96 (**+739**) |
| none | — | ppl 43.33 |

**STATUS** **FALSIFIED by our own measurements.** Already reset to `[UNKNOWN]`
in the notes. Phase E re-opens it as a structural comparison rather than a
guess-and-check.

---

## C8 — "The released artifact represents exactly the method in the paper"

**EVIDENCE**
- `[ARTIFACT]` It carries a per-channel rescale absorbed into the RMSNorms; the
  paper describes no such step and advertises the method as "bias-free and
  mask-free" (§3.2).
- `[ARTIFACT]` It leaves `q_proj`/`k_proj` unquantized, against §4.1.
- `[MEASURED]` It scores 35.256 where the paper reports 38.02 — **better than
  its own paper**, which is itself evidence the artifact and the Table 1 run are
  not the same procedure.

**STATUS** **ESTABLISHED FALSE.** This one is settled and in our favour as a
piece of knowledge: it means "reproduce the paper" and "reproduce the artifact"
are two different targets, and we have been measuring against both.

---

## C9 — "PTQTP's effective representation is 4.25 bpw"

**EVIDENCE** `[DERIVED]` from `[PAPER]` App. A.3 (2 bits per trit, unpacked →
4.000) plus Eq. 9 with two f16 scale vectors at the paper's own G=128
(2×16/128 = 0.250).

**COUNTER-EVIDENCE** `[PAPER]` The Tied paper quotes **4.125** for the same
free-scale two-plane serving format, which implies 256-element blocks
(2×16/256 = 0.125). Both are internally consistent; they differ in block size,
and PTQTP's text specifies G=128 for the *fit* without specifying the *storage*
block.

**STATUS** **SUPPORTED as one member of a family**, not as *the* number. Phase H
tabulates all of them with formulas.

---

## C10 — "PTQTP cannot provide an arithmetic advantage over Q4/Q8 on CPU"

**EVIDENCE**
- `[MEASURED]`, sibling: no target ISA has a sub-byte multiply-accumulate; NEON
  `SDOT`/`SMMLA`, AVX-512 VNNI `VPDPBUSD` and AMX are int8-lane. Our own
  `src/qmat.c` widens Q4 nibbles to int8 before the dot.
- `[PAPER]` Tied Trit-Planes confirms it from the other side: its nine-level
  codes are consumed *"via integer SIMD (sdot on NEON; vpshufb/vpsignb/vpdpbusd
  on AVX2/AVX-VNNI)"* — int8 lanes, one float multiply-add per block.
- `[PAPER]` PTQTP's own GPU kernel (Table 5) is **slower than GPTQ 4-bit** at
  batch 1.

**COUNTER-EVIDENCE** `[PAPER]` The same Tied paper measures **6.7% faster
decode** than `q4_k`. So the arithmetic claim and the throughput claim come
apart: no MAC advantage, but a real byte advantage that converts to time in a
bandwidth-bound regime.

**STATUS** **The arithmetic half is SUPPORTED. The "no CPU advantage" half is
FALSIFIED.** They must not be stated as one claim again. What ternary can offer
a CPU is *fewer weight bytes and possibly less unpack work* — never more MACs
per instruction.

---

## C11 — new, and it indicts our whole metric

**CLAIM (theirs, not ours)** `[PAPER]` Tied Trit-Planes reports that its tied fit
shows **higher weight-reconstruction error and worse perplexity** than the free
fit, while showing **no detected fidelity difference** against the reference API
on 5/5 fixtures and scoring **86 vs 84** on a 100-item MMLU subset — *"a measured
dissociation between proxy metrics and reference fidelity."*

**WHY IT MATTERS HERE** Every Gate A number we have produced is perplexity, and
we have run **zero** behavioural evals. This project's actual shipping gate is
`tools/eval/tool_calls.py`, where Qwen3-0.6B Q4_K_M scores 25/30. A +68%
perplexity result may or may not survive contact with that gate, in either
direction.

**STATUS** `[PAPER]`, and a standing caution on every conclusion in this study.
The tool-call and multilingual evals are now a required part of Gate A, not an
optional extra.

---

## Summary of status changes

| claim | was | now |
|---|---|---|
| C1 ALU-bound | stated broadly | **SUPPORTED, narrowly scoped** |
| C2 ternary can't help CPU | stated as conclusion | **WITHDRAWN** |
| C3 nothing survives <3 bits | universal | **RESTATED to the tested methods** |
| C4 0.6B is the fragile case | asserted, "monotone" | **NOT ESTABLISHED; our reading of the paper was wrong** |
| C5 PTQTP ≈ Q4_K_M size | stated flatly | **one configuration of several** |
| C6 q/k protected *because* sensitive | hedged already | observation `[ARTIFACT]`, motive `[UNKNOWN]` |
| C7 scale absorption explains the gap | proposed | **FALSIFIED, now `[UNKNOWN]`** |
| C8 artifact == paper method | assumed early | **ESTABLISHED FALSE** |
| C9 4.25 bpw | stated as the number | **one of a family** |
| C10 no CPU advantage | one claim | **split: arithmetic SUPPORTED, throughput FALSIFIED** |
| C11 perplexity is sufficient | never examined | **challenged by `[PAPER]`; behavioural eval now required** |
