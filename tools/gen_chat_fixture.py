"""Generate tests/fixtures/chat_tools.txt — HF apply_chat_template ground truth.

src/template.c writes ChatML out by hand instead of interpreting the Jinja
string in the GGUF, because shipping a Jinja engine to render six lines of
markup would be the largest dependency in the project. The price of that
decision is this file: the rendering has to be pinned against HF's, byte for
byte, or the model quietly sees a prompt it was never trained on.

Tool calling makes that sharper. The `# Tools` preamble, the spacing inside
each schema, the fact that tool RESULTS are <tool_response> blocks inside one
user turn rather than turns of their own — get any of it wrong and the model
still answers, just worse, and nothing in the output says why.

The cases here must stay in the same order as the ones in tests/test_tools.c.
Each is one line:

    <name>\\t<rendered prompt, with \\n \\t \\r \\\\ escaped>

    uv run --extra fixtures python gen_chat_fixture.py > ../tests/fixtures/chat_tools.txt
    uv run --extra fixtures python gen_chat_fixture.py ../reference/granite-4.0-350m \
        > ../tests/fixtures/chat_tools_granite.txt
"""

from __future__ import annotations

import os
import sys

from transformers import AutoTokenizer

DEFAULT_REF = os.path.join(os.path.dirname(__file__), "..", "reference", "qwen3-0.6b")

WEATHER = {
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
}

# No description, no parameters: both keys are optional and their absence must
# not leave a dangling comma in the rendered schema.
PING = {"type": "function", "function": {"name": "ping"}}

CASES = [
    ("plain", dict(
        messages=[{"role": "user", "content": "What is the weather in Verona?"}],
        tools=None, enable_thinking=False)),
    ("plain_think", dict(
        messages=[{"role": "user", "content": "Quanto fa 17+25?"}],
        tools=None, enable_thinking=True)),
    ("system", dict(
        messages=[{"role": "system", "content": "You are terse."},
                  {"role": "user", "content": "Hi"}],
        tools=None, enable_thinking=False)),
    ("tools_one", dict(
        messages=[{"role": "user", "content": "What is the weather in Verona?"}],
        tools=[WEATHER], enable_thinking=False)),
    ("tools_system", dict(
        messages=[{"role": "system", "content": "You are terse."},
                  {"role": "user", "content": "Weather in Verona?"}],
        tools=[WEATHER], enable_thinking=False)),
    ("tools_bare", dict(
        messages=[{"role": "user", "content": "ping please"}],
        tools=[PING], enable_thinking=True)),
    ("tools_two", dict(
        messages=[{"role": "user", "content": "Weather in Verona, then ping."}],
        tools=[WEATHER, PING], enable_thinking=True)),
    # The full round trip: the model called a function, we ran it, the result
    # goes back. This is the case that has to be right for a second turn to
    # make any sense at all.
    ("roundtrip", dict(
        messages=[
            {"role": "user", "content": "What is the weather in Verona?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Verona"}}}]},
            {"role": "tool", "content": '{"temp_c": 18, "sky": "clear"}'},
        ],
        tools=[WEATHER], enable_thinking=False)),
    # Two calls in one turn and two results: the results are ONE user turn with
    # two <tool_response> blocks, not two turns.
    ("parallel", dict(
        messages=[
            {"role": "user", "content": "Weather in Verona and Trento?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Verona"}}},
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Trento"}}}]},
            {"role": "tool", "content": '{"temp_c": 18}'},
            {"role": "tool", "content": '{"temp_c": 15}'},
        ],
        tools=[WEATHER], enable_thinking=False)),
    # Text and a call in the same assistant turn: the newline between them is
    # emitted only because there is text in front of it.
    ("text_and_call", dict(
        messages=[
            {"role": "user", "content": "Check Verona."},
            {"role": "assistant", "content": "Let me look that up.", "tool_calls": [
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Verona"}}}]},
            {"role": "tool", "content": '{"temp_c": 18}'},
        ],
        tools=[WEATHER], enable_thinking=False)),
]


def escape(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
             .replace("\t", "\\t").replace("\r", "\\r"))


def main() -> int:
    # Which family to render. The cases are the same either way — that is the
    # point of the fixture: two templates, one set of situations, and the C
    # renderer has to match both byte for byte.
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REF
    tok = AutoTokenizer.from_pretrained(ref)
    for name, kw in CASES:
        extra = {}
        if "enable_thinking" in (tok.chat_template or ""):
            extra["enable_thinking"] = kw["enable_thinking"]
        text = tok.apply_chat_template(
            kw["messages"],
            tools=kw["tools"],
            add_generation_prompt=True,
            tokenize=False,
            **extra,
        )
        print(f"{name}\t{escape(text)}")
    print(f"{len(CASES)} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
