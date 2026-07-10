#!/usr/bin/env python3
"""Regression gate: every observed MiniMax tool-markup drift flavor must be
DETECTED (never ships as prose) and, where the call is unambiguous, RECOVERED.

Run:  ~/mlx-vlm064-env/bin/python3.14 ops/test_tool_parse_flavors.py
Add a case every time a new flavor leaks; this file is the whack-a-mole ledger.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "sharded_server.py"
NS = "]<]minimax[>["
START = NS + "<tool_call>"


def load_funcs():
    src = SRC.read_text()
    ns = {"re": re, "json": json}
    import logging
    ns["logger"] = logging.getLogger("flavors")
    ns["_tool_call_markers"] = lambda tm: (START, NS + "</tool_call>")
    ns["_tool_name_map_from_schema"] = lambda tools: {
        t["function"]["name"].lower(): t["function"]["name"] for t in (tools or [])
    }
    ns["_canonical_tool_name"] = lambda raw, m: m.get((raw or "").lower())
    ns["_canonicalize_tool_argument_keys"] = lambda obj, tools, name: obj
    ns["_tool_schema_expects_arguments"] = lambda tools, name: True
    ns["_openai_tool_call"] = lambda name, args, i: {
        "function": {"name": name, "arguments": args}
    }
    for fn in ("_json_balanced_end", "_looks_like_raw_tool_fragment",
               "_recover_bare_name_tool_calls"):
        i = src.find(f"def {fn}(")
        assert i >= 0, fn
        j = src.find("\ndef ", i + 1)
        exec(compile(src[i:j], str(SRC), "exec"), ns)
    return ns


TOOLS = [{"type": "function", "function": {"name": n}}
         for n in ("Bash", "Write", "terminal", "write_file")]

CASES = [
    # (label, text, expect_detect, expect_recovered_name)
    ("bare-name+json (zcode 07-10a)",
     f'Let me check.{NS} Bash {{"command":"ls -la","description":"list"}}',
     True, "Bash"),
    ("self-describing json (zcode 07-10b)",
     f'Writing the game.{NS} {{"name": "Write", "arguments": {{"file_path": "/tmp/h.html", "content": "<html>x</html>"}}}}',
     True, "Write"),
    ("self-describing, input key",
     f'{NS}{{"name": "terminal", "input": {{"command": "pwd"}}}}',
     True, "terminal"),
    ("unterminated tag marker (hermes 07-10)",
     f"plan done</mm:think>{START}",
     True, None),
    ("classic tag fragment",
     f'{START} {NS}<invoke name="Bash">',
     True, None),
    ("raw newlines in self-describing json",
     f'{NS} {{"name": "write_file", "arguments": {{"path": "a.md", "content": "# T\nline\n"}}}}',
     True, "write_file"),
    ("plain prose (must NOT trip)",
     "The ocean is salty because rivers carry dissolved minerals.",
     False, None),
    ("prose quoting marker w/o call shape (accepted rare FP=False)",
     f"The internal syntax {NS} is used by the wire protocol.",
     False, None),
]


def main() -> int:
    ns = load_funcs()
    detect = ns["_looks_like_raw_tool_fragment"]
    recover = ns["_recover_bare_name_tool_calls"]
    failures = 0
    for label, text, want_detect, want_name in CASES:
        got_detect = bool(detect(text, object()))
        calls = recover(text, object(), TOOLS)
        got_name = (calls[0]["function"]["name"] if calls else None)
        ok = (got_detect == want_detect) and (want_name is None or got_name == want_name)
        print(f"{'PASS' if ok else 'FAIL'}  {label}: detect={got_detect} "
              f"recovered={got_name}")
        failures += (not ok)
    print(f"\n{len(CASES)-failures}/{len(CASES)} flavors green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
