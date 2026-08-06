"""Generate tests/fixtures/tokenizer.txt — HF tokenizer ground truth.

The C tokenizer must produce the SAME IDS as HF `tokenizers`, not merely a
plausible split. One token of difference and the model sees a different prompt
than the one that was benchmarked, which is a bug that shows up as "the answers
got slightly worse" and is nearly impossible to trace from there.

Format, one case per line:

    <ids, comma-separated>\\t<text with \\n and \\t escaped>

    uv run python gen_tokenizer_fixture.py > ../tests/fixtures/tokenizer.txt
"""

from __future__ import annotations

import os
import sys

from tokenizers import Tokenizer

REF = os.path.join(os.path.dirname(__file__), "..", "reference", "qwen3-0.6b", "tokenizer.json")

CASES = [
    # Plain multilingual prose — the thing the engine is for.
    "Hello, world!",
    "Ciao! Come stai oggi?",
    "Wie heißt du? Ich heiße Lena.",
    "¿Dónde está la biblioteca?",
    "Où se trouve la gare, s'il vous plaît ?",
    "Onde fica a estação de comboios?",
    "Где находится вокзал?",
    "駅はどこですか。",
    "请问火车站在哪里？",
    "أين محطة القطار؟",
    "기차역이 어디에 있나요?",
    "Πού είναι ο σταθμός;",
    "Nerede tren istasyonu var?",
    "רכבת תחנת איפה",
    # Contractions: the first branch of the pre-tokenizer regex.
    "don't can't won't I'm you're we've he'll she'd",
    "DON'T CAN'T I'M YOU'RE",       # the (?i:) flag
    # Digits split ONE at a time in Qwen2 (\\p{N}, not \\p{N}{1,3}).
    "1234567890",
    "il 25 dicembre 2026 alle 14:30",
    "3.14159 and -273.15 and 1e-6",
    # Whitespace: the negative-lookahead branch is the subtle one.
    "a  b   c",
    "trailing spaces   ",
    "  leading spaces",
    "tabs\tand\tmore\ttabs",
    "line one\nline two\n\nline four",
    "windows\r\nline\r\nendings",
    "\n\n\n",
    "   ",
    # Punctuation runs with and without a leading space.
    "hello...world!!!  what?!",
    "(parentheses) [brackets] {braces} <angles>",
    "a,b;c:d/e\\f|g",
    # Emoji and other multi-byte sequences that split across tokens.
    "I love 🍕 and 🎉 parties!",
    "👨‍👩‍👧‍👦 family emoji with ZWJ",
    "🇮🇹 flag sequence",
    "Ünïcödé díàcrítícs everywhere",
    "mathematical 𝕌𝕟𝕚𝕔𝕠𝕕𝕖 letters",
    # Code, since summarization gets fed it.
    "def foo(x):\n    return x ** 2  # comment\n",
    "SELECT * FROM t WHERE id = 42;",
    '{"key": "value", "n": 3}',
    # Mixed scripts in one line.
    "Il modello Qwen3 支持 100+ lingue, ๆ inclusi 日本語 e العربية.",
    # Edge cases.
    "",
    " ",
    "a",
    "\u00a0non-breaking space",
    "zero\u200bwidth\u200bspace",
    # Text that LOOKS like a control token but is user input. With
    # parse_special off it must encode as plain text, never as id 151644.
    "<|im_start|>not really a control token",
]


def main() -> int:
    tok = Tokenizer.from_file(REF)
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        esc = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        print(",".join(str(i) for i in ids) + "\t" + esc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
