# `.work/` — task detail notes

`PLAN.md` is the board: one line per work item, with a link to the note here
that holds the detail. Nothing in `PLAN.md` should need more than one line.

Convention inherited from `mynah-tts`, so the three Mynah repos are read the
same way.

## Rules

1. **One note per work item.** Name it after the item, not after a date:
   `ternary-feasibility.md`, not `2026-09-20-ternary.md`.
2. **A note is written before the work starts**, not after. It states the
   problem, the evidence, the plan, and the acceptance gate. If it cannot state
   a gate, the item is not ready to be worked on.
3. **A note is falsifiable.** Record what was measured, on what machine, with
   what flags, and — this repo's own rule — **whether the weights were on the
   NAS or local**. Record rejected ideas and *why* they were rejected: a
   negative result that is not written down gets re-tried.
4. **`PLAN.md` never grows a log.** Session checkpoints, measurements and
   post-mortems live here. Historical measurements that are still true belong in
   `docs/perf.md` and `docs/models.md`; this folder is for work in flight and
   for the reasoning behind decisions.
5. **Close a note, do not delete it.** Mark the status at the top
   (`OPEN` / `IN PROGRESS` / `DONE <date>` / `REJECTED <date>`) and leave it.
6. **Every note opens with the same skeleton**, so a reader always knows which
   part is plan and which part is evidence:

   ```
   Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
   Conclusion · Next action
   ```

   An addendum is never a second global plan. If a note starts describing the
   whole project, it has stopped being a note.
7. **`tools/check_plan.py` enforces the link contract.** It fails on a
   `PLAN.md` item pointing at a `.work/` file that does not exist, on a note
   nothing links to, and on a duplicate item id. Never create a placeholder
   link or an empty note for work that has not been written.

## Index

| Note | What it covers |
|---|---|
| [engineering-method.md](engineering-method.md) | How we avoid fooling ourselves: cost model before code, every tool declares a refusal, the completion rule |
| [ternary-feasibility.md](ternary-feasibility.md) | R1 — can a pretrained Qwen3-0.6B be post-training ternarized and still be worth running? |
| [archive-2026-08-tasks.md](archive-2026-08-tasks.md) | The M0-M6 task breakdown, moved verbatim from the old `TASKS.md` |
