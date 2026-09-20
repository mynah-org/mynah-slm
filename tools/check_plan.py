#!/usr/bin/env python3
"""Plan integrity check: PLAN.md is a board, .work/ holds the detail.

The convention is documented in .work/README.md. This script is the part that
refuses, so the convention cannot rot quietly:

  * a PLAN.md link into .work/ that points at a file which does not exist;
  * a .work/ note that nothing on the board links to;
  * a note missing its status marker;
  * a duplicate work-item id on the board.

It prints every problem it finds and exits 1, or prints one OK line and exits 0.
No dependencies: it runs under a bare python3, so `make test` can call it on a
machine that never ran `uv sync`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "PLAN.md"
WORK = REPO / ".work"

# `[text](.work/foo.md)` and `[text](foo.md)` from inside .work/
LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:#[^)]*)?\)")
# A board line: "- [x] **R1** — ..." or "- [ ] M0-M6 → ..."
ITEM_RE = re.compile(r"^\s*-\s*\[[ x~-]\]\s*(?:\*\*)?([A-Z]\d+[a-z]?(?:-\d+)?)(?:\*\*)?\b")
STATUS_RE = re.compile(
    r"^Status:\s*\*\*(OPEN|IN PROGRESS|DONE|REJECTED|REFERENCE|ARCHIVE)\b", re.M
)


def main() -> int:
    problems: list[str] = []

    if not PLAN.is_file():
        print("check_plan: PLAN.md is missing", file=sys.stderr)
        return 1
    if not WORK.is_dir():
        print("check_plan: .work/ is missing", file=sys.stderr)
        return 1

    plan = PLAN.read_text(encoding="utf-8")

    # 1. Every .work/ link on the board resolves.
    linked: set[str] = set()
    for target in LINK_RE.findall(plan):
        if not target.startswith(".work/"):
            continue
        name = target[len(".work/") :]
        linked.add(name)
        if not (WORK / name).is_file():
            problems.append(f"PLAN.md links to .work/{name}, which does not exist")

    # 2. Every note is reachable from the board, directly or from README's index.
    readme = WORK / "README.md"
    index_links = set()
    if readme.is_file():
        index_links = {
            t for t in LINK_RE.findall(readme.read_text(encoding="utf-8")) if "/" not in t
        }
    else:
        problems.append(".work/README.md is missing")

    for note in sorted(WORK.glob("*.md")):
        if note.name == "README.md":
            continue
        if note.name not in linked and note.name not in index_links:
            problems.append(
                f".work/{note.name} is linked from neither PLAN.md nor .work/README.md"
            )
        # 3. Every note declares a status.
        if not STATUS_RE.search(note.read_text(encoding="utf-8")):
            problems.append(
                f".work/{note.name} has no 'Status: **...**' marker "
                "(OPEN / IN PROGRESS / DONE <date> / REJECTED <date> / REFERENCE / ARCHIVE)"
            )

    # 4. No duplicate work-item ids on the board.
    seen: dict[str, int] = {}
    for lineno, line in enumerate(plan.splitlines(), 1):
        m = ITEM_RE.match(line)
        if not m:
            continue
        item = m.group(1)
        if item in seen:
            problems.append(
                f"PLAN.md:{lineno}: duplicate work-item id {item!r} "
                f"(first seen at line {seen[item]})"
            )
        else:
            seen[item] = lineno

    if problems:
        print("check_plan: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    notes = len(list(WORK.glob("*.md"))) - 1
    print(f"check_plan: OK — {len(seen)} board items, {notes} notes, all links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
