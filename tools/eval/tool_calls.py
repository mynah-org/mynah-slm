"""Does the model actually call the tools? Measured, per language.

The engine can render the schemas and parse a call back out; that is what
tests/test_tools.c pins. Neither of those says whether a 0.6B model DECIDES
correctly — and the failure that matters is not malformed JSON, it is a model
that answers the weather question out of its own head instead of calling the
function it was handed. That failure looks like a perfectly good answer.

So every case here scores three separate things:

  decision  did it call a tool when it should have, and stay quiet when it
            should not? A model that calls a function to say hello is as
            broken as one that never calls anything.
  name      the right function.
  arguments the required ones present, with the expected values.

Negative cases are half the suite on purpose. Run it per language: a model can
be reliable in English and hallucinate in Italian, and one aggregate number
hides exactly that.

    uv run python -m eval.tool_calls ../models-local/Qwen3-0.6B-Q4_K_M.gguf
    uv run python -m eval.tool_calls MODEL --lang it --think on
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient address"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a countdown timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "Duration in minutes"},
                    "label": {"type": "string"},
                },
                "required": ["minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query and return results",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

# expect=None means: answer in words, do NOT call anything.
CASES = [
    # ── English ───────────────────────────────────────────────────────────
    ("en", "What is the weather in Verona?", ("get_weather", {"city": "Verona"})),
    ("en", "Is it raining in Berlin right now?", ("get_weather", {"city": "Berlin"})),
    ("en", "Set a timer for 10 minutes.", ("set_timer", {"minutes": 10})),
    ("en", "Email anna@example.com with the subject Lunch and the body See you at one.",
     ("send_email", {"to": "anna@example.com"})),
    ("en", "Search the web for the best pizza in Naples.", ("search_web", None)),
    ("en", "Hello! How are you today?", None),
    ("en", "Write a haiku about the sea.", None),
    ("en", "What is 17 plus 25?", None),
    # ── Italian ───────────────────────────────────────────────────────────
    ("it", "Che tempo fa a Verona?", ("get_weather", {"city": "Verona"})),
    ("it", "Sta piovendo a Berlino adesso?", ("get_weather", {"city": "Berlino"})),
    ("it", "Imposta un timer di 10 minuti.", ("set_timer", {"minutes": 10})),
    ("it", "Manda una mail a anna@example.com con oggetto Pranzo e testo Ci vediamo all'una.",
     ("send_email", {"to": "anna@example.com"})),
    ("it", "Cerca sul web la migliore pizza di Napoli.", ("search_web", None)),
    ("it", "Ciao! Come stai oggi?", None),
    ("it", "Scrivi un haiku sul mare.", None),
    ("it", "Quanto fa 17 più 25?", None),
    # ── German ────────────────────────────────────────────────────────────
    ("de", "Wie ist das Wetter in Verona?", ("get_weather", {"city": "Verona"})),
    ("de", "Stelle einen Timer auf 10 Minuten.", ("set_timer", {"minutes": 10})),
    ("de", "Suche im Internet nach der besten Pizza in Neapel.", ("search_web", None)),
    ("de", "Hallo! Wie geht es dir?", None),
    # ── French ────────────────────────────────────────────────────────────
    ("fr", "Quel temps fait-il à Vérone ?", ("get_weather", {"city": "Vérone"})),
    ("fr", "Mets un minuteur de 10 minutes.", ("set_timer", {"minutes": 10})),
    ("fr", "Cherche sur le web la meilleure pizza de Naples.", ("search_web", None)),
    ("fr", "Bonjour ! Comment vas-tu ?", None),
    # ── Spanish ───────────────────────────────────────────────────────────
    ("es", "¿Qué tiempo hace en Verona?", ("get_weather", {"city": "Verona"})),
    ("es", "Pon un temporizador de 10 minutos.", ("set_timer", {"minutes": 10})),
    ("es", "Busca en la web la mejor pizza de Nápoles.", ("search_web", None)),
    ("es", "¡Hola! ¿Cómo estás?", None),
    # ── Portuguese / Dutch, one probe each ────────────────────────────────
    ("pt", "Como está o tempo em Verona?", ("get_weather", {"city": "Verona"})),
    ("nl", "Wat voor weer is het in Verona?", ("get_weather", {"city": "Verona"})),
]


def run_case(binary, model, tools_path, prompt, think, max_new, threads):
    cmd = [binary, "run", "-m", model, "--tools", tools_path, "-p", prompt,
           "-n", str(max_new), "--temp", "0", "--think", think, "--no-stream"]
    if threads:
        cmd += ["-t", str(threads)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    dt = time.monotonic() - t0
    # A crashed CLI produces no calls, which would otherwise be scored as
    # "answered in words" — a silent zero that looks like a model result.
    if p.returncode != 0:
        raise SystemExit(f"mynah-slm exited {p.returncode}:\n{p.stderr.strip()}")

    calls, text = [], []
    for line in p.stdout.splitlines():
        if line.startswith('{"tool_calls":'):
            try:
                calls = json.loads(line)["tool_calls"]
            except json.JSONDecodeError:
                calls = [{"broken": line}]
        else:
            text.append(line)
    return calls, "\n".join(text).strip(), p.stderr, dt


def score(case, calls):
    """(verdict, detail). verdict is one of pass / no_call / spurious / wrong."""
    lang, prompt, expect = case

    if expect is None:
        if calls:
            return "spurious", f"called {calls[0]['function']['name']} for a chat turn"
        return "pass", ""

    want_name, want_args = expect
    if not calls:
        return "no_call", "answered in words instead of calling"

    fn = calls[0].get("function", {})
    got_name = fn.get("name", "?")
    if got_name != want_name:
        return "wrong", f"called {got_name}, expected {want_name}"

    try:
        args = json.loads(fn.get("arguments", "{}"))
    except json.JSONDecodeError:
        return "wrong", f"arguments are not JSON: {fn.get('arguments')!r}"
    if not isinstance(args, dict):
        return "wrong", f"arguments are not an object: {args!r}"

    for key in (want_args or {}):
        if key not in args:
            return "wrong", f"missing argument {key!r} in {args}"
        want, got = want_args[key], args[key]
        if str(want).strip().lower() != str(got).strip().lower():
            return "wrong", f"{key}={got!r}, expected {want!r}"
    return "pass", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--bin", default=os.path.join(ROOT, "mynah-slm"))
    ap.add_argument("--think", default="off", choices=["off", "low", "on"])
    ap.add_argument("--lang", default=None, help="only this language")
    ap.add_argument("-n", "--max-new", type=int, default=256)
    ap.add_argument("-t", "--threads", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full transcript here")
    args = ap.parse_args()

    # Resolved here because the CLI runs with cwd=ROOT while this script is
    # usually invoked from tools/: a relative model path would point at a
    # different directory in the child and fail there, not here.
    args.model = os.path.abspath(args.model)
    args.bin = os.path.abspath(args.bin)

    if not os.path.exists(args.model):
        print(f"no model at {args.model} (scripts/use_model.sh)", file=sys.stderr)
        return 77

    cases = [c for c in CASES if args.lang is None or c[0] == args.lang]

    fd, tools_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(TOOLS, f)

    verdicts, rows = {}, []
    try:
        print(f"{len(cases)} cases | think={args.think} | {os.path.basename(args.model)}\n")
        for lang, prompt, expect in cases:
            calls, text, stderr, dt = run_case(
                args.bin, args.model, tools_path, prompt, args.think,
                args.max_new, args.threads)
            verdict, detail = score((lang, prompt, expect), calls)
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            malformed = "malformed tool call" in stderr

            mark = {"pass": "ok  ", "no_call": "MISS", "spurious": "SPUR",
                    "wrong": "WRNG"}[verdict]
            print(f"{mark} [{lang}] {prompt[:52]:<52} {dt:5.1f}s"
                  + (f"  <- {detail}" if detail else ""))
            if malformed:
                print("       (a tool call block did not parse)")

            rows.append({"lang": lang, "prompt": prompt,
                         "expected": expect and expect[0], "verdict": verdict,
                         "detail": detail, "calls": calls, "text": text,
                         "malformed": malformed, "seconds": round(dt, 2)})
    finally:
        os.unlink(tools_path)

    total = len(rows)
    ok = verdicts.get("pass", 0)
    print(f"\n{ok}/{total} correct ({100.0 * ok / total:.0f}%)")
    for k in ("no_call", "spurious", "wrong"):
        if verdicts.get(k):
            label = {"no_call": "answered instead of calling",
                     "spurious": "called a tool for a chat turn",
                     "wrong": "wrong function or arguments"}[k]
            print(f"  {verdicts[k]:2d}  {label}")

    print("\n  lang   correct")
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        good = sum(1 for r in sub if r["verdict"] == "pass")
        print(f"  {lang:<5}  {good}/{len(sub)}")

    n_malformed = sum(1 for r in rows if r["malformed"])
    print(f"\n  JSON validity: {total - n_malformed}/{total} turns parsed cleanly")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"model": args.model, "think": args.think, "rows": rows}, f, indent=2)
        print(f"  transcript: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
