# The method — how we avoid fooling ourselves

Status: **REFERENCE** — read before starting anything, not a task.

Lineage: qwen-tts `ENGINEERING.md` / `docs/ENGINEERING-METHOD.md` (which declares
its own lineage: Abrash's *Graphics Programming Black Book*), distilled in
`mynah-tts/.work/engineering-method.md`, re-framed here for an SLM engine. The
incidents cited from sibling repos are kept with their attribution, because a
rule without the incident that produced it gets argued away.

Almost none of this is code, which is why it is the cheapest thing in the
project to adopt.

---

## 1. The four opening lines

> **Do not optimize the code. Optimize the work.**
> **Do not trust expected behaviour. Observe the machine.**
> **Do not celebrate a faster component until the system is faster.**
> **Every unexplained millisecond is an engineering question.**

With the disclaimer that prevents the usual misreading:

> This is not a low-level-programming aesthetic. Hand-written SIMD is not the
> point, and preferring it for its own sake would be the opposite of the lesson.
> The rule exists to make us **more suspicious, especially of results we like.**

---

## 2. The organising principle

> The agent must not remember the project; the repository must make it impossible
> to forget. `PLAN.md` says what remains, `AGENTS.md` says how we work, the
> `.work/` notes hold a task's detail, and **the program itself proves which
> path it is executing.**

That last clause is load-bearing. It is why `mynah-slm inspect` exists, why the
speed line is mandatory on every delivery mode, and why the kernel A/B in
`tests/bench_matvec.c` is *interleaved in one process* instead of two runs.

### Rules, with the reason attached

**Separate plan from evidence.** Every note opens with the fixed skeleton:
`Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
Conclusion · Next action`.

**Plan integrity is checked by a script.** `tools/check_plan.py` fails on a
missing `.work/` file, a duplicate item id, a done item pointing at a file that
does not exist, or a note naming an item the plan does not have. *"Never create
placeholder links or empty notes for work that has not been written."*

**Backend claims require a matrix.** Never call a change "cross-platform", "ARM"
or "x86" without reading its gates. Mandatory classification: **A** common
runtime · **B** common design, backend-specific implementation · **C**
backend/ISA-specific.

**A benchmark is invalid until dispatch is proven.**

> Never infer the active kernel from a profile filename, a requested env flag or
> the build target name. The strongest evidence is in-process: the resolved
> table at startup and the kernel census after warm-up, compared against an
> expected/allowed/forbidden manifest.
>
> An explicit request is a request, not a capability. If an explicitly requested
> path differs from the resolved one: **stop, do not run the benchmark. A
> fallback run is never reported as a measurement of the requested path.**

This repo has already been bitten by exactly this, twice:

- **Rosetta lies about CPUID.** It executes AVX2 and does not advertise it, so
  ingot's runtime dispatch picked scalar and the whole x86 suite passed without
  running a single AVX2 kernel. `INGOT_CAPS_ASSUME=avx2` trusts the build
  instead; `make test-x86-rosetta` sets it. Before believing any x86 number
  taken here, check that the path was actually taken.
- **The kernel A/B disagreed with itself by 80%** across two separate runs, on
  tensors neither side touches. The fix was not a better timer: it was to
  interleave A and B *inside one process* and to keep the untouched tensors in
  the report as a control. A control that lands on 1.00x is what makes the rest
  of the table readable.

**Fallbacks must be visible.** Every significant optimised operation reports
which implementation ran, from at least: ours (SIMD) · ours (scalar) · ingot ·
BLAS · unsupported. *"A silent fallback in a qualification run is a failure."*

**Never benchmark a contradiction.** Whenever a contradiction is mechanically
detectable, add a preflight gate that refuses the run. Ours, concretely:

- a decode t/s measured with the weights on the **NAS** and reported as a
  product number — 17.0 s vs 7.5 s on the same file is not a tolerance, it is a
  different experiment;
- a "Q4_K_M is 4-bit" claim — that file is 5.24 bits/weight and `inspect` will
  say so;
- a prefill t/s quoted where a decode t/s was asked for.

**Explicit terminology, no bare numbers.** `decode tok/s` is the headline and it
is memory-bandwidth bound; `prefill tok/s` is a different number and must never
stand in for it. `TTFT` is measured at the first token, never reconstructed at
the end. Every benchmark line states quant, threads, machine, and NAS-or-local.

**No opportunistic scope expansion.** While executing X do not start Y, do not
change numerical precision, do not change production defaults, do not build a
new profiling framework.

**Numerical changes are separate work.** *"An optimisation that changes the
arithmetic or the output is not a structural optimisation. It needs its own
quality qualification and is never promoted on performance numbers alone."*
Every sub-4-bit candidate in this repo is gated by a **measured** multilingual +
tool-call regression, never by vibes. That is the rule that killed `Q2_K`
(ppl 26400, word salad) and that put `Q3_K_M` on the shelf as a bad trade rather
than a failure.

**What is committed is what was built.** Canonical evidence comes from a clean
committed tree; a dirty-tree binary is labelled NON-QUALIFYING.

**One implementation owner; parallel agents are evidence suppliers.**
Canonical state is a **HASH**, never "roughly the latest branch". Before
delegating, emit `CANONICAL_HEAD=$(git rev-parse HEAD)` and require the analyst
to prove `HEAD == CANONICAL_HEAD`, clean tree, zero source diff. Analysts return
EVIDENCE / INTERPRETATION / CONFIDENCE / CONTRADICTIONS / RECOMMENDATION. **Read
the evidence before the recommendation. A contradiction an analyst surfaces is
priority evidence.** (mynah-tts, 2026-09-06: two trees evolved independently for
a day; one held ~60 commits the user's branch did not have, the other ~1261
uncommitted lines any fast-forward would have destroyed. Nothing was lost and
the day was lost anyway.)

**Completion rule.** A task is complete only when the final report contains:
`WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS
UNKNOWN · VERDICT: PROMOTE / KEEP / INCONCLUSIVE / REJECT`.

---

## 3. Cost model before code

Every optimisation gets this written down *before* any is written:

| field | |
|---|---|
| current cost | measured, not assumed |
| suspected cause | |
| proposed transformation | |
| **maximum plausible saving** | if it is too small for the goal, **do not optimise** |
| new work introduced | |
| risk | |
| **the smallest experiment capable of killing the idea** | |

And before every experiment: *"which control would make my preferred explanation
look stupid if it were wrong?"*

Supporting rules:

- **A label is not a causal explanation.** `prefill`, `decode`, `GEMV`,
  `dequant` — *"it is another box to open."*
- *"A ten-line change that removes 15 ms is better engineering than a
  thousand-line subsystem that removes 17."*
- The claim chain: `microbenchmark → tensor → layer → decode step → CLI run →
  server under load → quality`. **Never skip a level.**
- *"Do not spend twenty minutes proving what a twenty-second component
  measurement could refuse."*
- The **engineering ledger**: FACT · HYPOTHESIS · TEST · RESULT · DECISION ·
  CLAIM SCOPE. *"Rejected ideas are evidence. Do not silently rediscover and
  re-run them."*
- **Measure differences, not absolutes**, when the number carries system noise
  (page cache, thermal drift, warm-up).

The repo's own worked example of a cost model applied honestly is the **Q6_K
kernel that was written, measured, upstreamed, then deleted**: a tie on ARM, and
once ingot had the AVX2 twin, ingot's version measured *faster* than ours. A
kernel with no per-call specialization belongs upstream. That is rule 4 working,
not an exception to it.

---

## 4. Every tool declares a refusal

The most copyable property in the whole toolchain, and it needs no kernel work:

- a census exits non-zero if any operation resolved UNKNOWN, or if a feature
  resolved ON whose class never executed;
- a roofline returns `ROOF UNKNOWN` (nothing of that kind measured) or
  `NOT COMPARABLE` (different residency) instead of dividing;
- a doctor prints `[UNKNOWN]` — *"nothing here supports a number, and the doctor
  says so instead"* — and labels every value `[MEASURED]` / `[CACHED]` /
  `[PREDICTED]` (±10-25%) / `[UNKNOWN]`;
- the dispatch gate aborts on a resolved fallback, and flags `SUSPICIOUS` =
  expected ON, compiled, supported, resolved OFF with no explicit env — *a
  fallback nobody asked for*.

**This repo's mandatory refusal**: anything that reads `models/` must fail with a
clear *"is the NAS mounted?"*, never a raw `FileNotFoundError`. The SMB mount has
gone into `Authentication error` — reads failing while `mount` still lists it —
three times in one session.

The sentence that generated the whole toolchain:

> A fast kernel does not make an efficient engine.

---

## 5. Execution discipline for tests and benchmarks

1. Run exactly one test or benchmark process at a time. No parallel tool calls,
   no background jobs, no `make -j`, no concurrent A/B for latency, throughput,
   memory, sanitizer or correctness measurements.
2. Distinguish process count from worker-thread count. Report both. Default a
   performance investigation to one process and an explicit thread count; sweep
   threads only as a separate, explicitly labelled experiment.
3. Change one variable per A/B. Identical binary, model file, quant, prompt,
   seed, sampler, thread count, environment, warm-ups and run count. Run A and B
   sequentially on the same machine. (Repo rule 9.)
4. At least two warm-ups and five measured runs for an accepted result. Report
   the median and the visible spread; never fold model load or cold page-in into
   a decode t/s.
5. Treat exploratory or contended numbers as invalid *immediately*. Do not
   quote, average, document or justify a change with them.
6. Verify correctness before calling a speedup: parity against the oracle at the
   declared per-stage tolerance, not an md5 and not "looks right".
7. Accept an optimisation only when it beats the declared noise floor (3%),
   survives the smallest relevant tests, and its complexity is justified. Revert
   failed experiments completely.
8. After interrupting a command, confirm the child processes exited. Never leave
   a benchmark or a sanitizer eating the user's machine.
9. **Never busy-poll a long run.** Launch it in the background and wait for the
   completion notification; re-reading its output file in a loop is the same
   thing as `sleep`+`cat` and costs the same. If active monitoring is genuinely
   needed, agree the cadence first and use one long-interval waiter.
10. Communicate like an owner: measured bottleneck, hypothesis, change, exact
    evidence, correctness result, remaining risk, next decision. Be precise
    about what was *not* tested.

---

## 6. Suspect the environment before the code

When the symptom is absurd — a tensor that is present but reads NULL on one
side, a binary behaving differently from yesterday with no relevant commit — the
build and the environment are cheaper to exclude than the code:

- `rm -rf build && make` costs minutes and closes an entire class of hypothesis;
- **make the program say where it dies.** One line of targeted diagnostic beats
  three bisects, narrows the field from a module to a line, and it keeps;
- **check that a green gate actually exercises the path you care about.** A test
  that passes while the main function is broken is worse than no test, because
  it authorises writing PASS into a commit.

**Apple ships GNU Make 3.81, whose file-time comparison is whole seconds**
(mynah-tts, 2026-09-14). An edit landing in the same second as the object's last
build is not newer by make's reckoning, is silently skipped, and the binary keeps
the old code with the new source on disk. The tighter the edit-build-test loop,
the likelier it is. It cuts both ways and the second way is worse: a stale object
can *pass* a test that should fail, so a mutation test may have run the unmutated
binary. Therefore: any gate that decides something gets a `make clean` first, or
at minimum an `rm` of the objects under test; and a mutation test is evidence
only if the mutated build is *observed* to differ.

---

## 7. The asymmetry checklist — nine classes to grep for

Applied here to ARM-vs-x86 and to ours-vs-ingot:

1. feature implemented but **wrong default/gate**;
2. **an extra conversion/transpose/repack present on only one architecture**;
3. automatic on one, behind an env flag on the other;
4. fused on one, several calls on the other;
5. different thread-pool / barrier behaviour;
6. different scratch allocation / copy / gather / scatter;
7. persistent representation on one vs per-call on the other;
8. different batch/shape thresholds;
9. **a generic fallback silently winning dispatch against the better kernel.**

Where to look: every `*_available()` predicate, every gate row with an *opt-in*
env instead of an opt-out, every `#if defined(__ARM_NEON)` with no `__AVX2__`
sibling, and every ISA-specific call whose counterpart is a **different
function** rather than the same one.

Class 9 is not hypothetical here: before 2026-08-07, `Q8_0` had **no vector path
on either architecture** in ingot, and `Q2_K`/`Q3_K` had no fused matvec at all.
Nothing reported a fallback, because nothing was asked to.

---

## 8. Where the time goes in a *small* model

The single most relevant number inherited from the sibling repos: activation
preparation as a share of `prep + compute` for one int8 projection.

| | B1 | B2 | B4 |
|---|---|---|---|
| large model, one projection | 28.4% | 42.8% | 61.6% |
| **small model, same projection** | **52.8%** | **70.3%** | **79.5%** |

**The smaller the model, the more of the time is preparation rather than
arithmetic.** mynah-slm's targets are 0.35B-0.6B. Expect every exotic weight
format to be gated by activation preparation and by the scratch round-trip, not
by the dot product — which is exactly why rule 4 says new kernels live in `src/`
where they can be fused, and why *"we keep our Q4_K because of the cross-row
`SUM x` hoist, which ingot's per-call API cannot do."*
