#!/usr/bin/env python3
"""Smoke-check the default MiniMax-M3 standard tool path.

This intentionally runs with MLX_M3_TOOL_COMPAT_OVERLAY=0. It verifies the
production behavior: native MiniMax-M3 XML tool calls are parsed, arguments are
shaped to the exact OpenAI tool schema, and legacy pseudo-tool recovery remains
disabled.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLX_M3_TOOL_COMPAT_OVERLAY", "0")
os.environ.setdefault("MLX_M3_TOOL_SYSTEM_HINT", "0")

import mlx_vlm.tool_parsers.minimax_m3 as minimax_m3

from sharded_server import (
    TOOL_COMPAT_OVERLAY,
    _parse_tool_calls,
    _tool_call_complete_for_stop,
    _validate_outgoing_tool_calls,
)


NS = "]<]minimax[>["


def write_tool(properties):
    return [{
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }]


def test_native_claude_style_write_call_with_camel_schema():
    tools = write_tool({
        "filePath": {"type": "string"},
        "content": {"type": "string"},
    })
    text = (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="Write">{NS}<content>Test file created on 2026-07-04.\n'
        f"{NS}</content>{NS}<filePath>~/Desktop/test.txt"
        f"{NS}</filePath>{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert remaining == "", remaining
    assert len(validated) == 1, validated
    fn = validated[0]["function"]
    assert fn["name"] == "Write", fn
    args = json.loads(fn["arguments"])
    assert args == {
        "filePath": "~/Desktop/test.txt",
        "content": "Test file created on 2026-07-04.",
    }, args
    assert _tool_call_complete_for_stop(text, minimax_m3, tools)


def test_native_claude_style_write_call_maps_to_snake_schema():
    tools = write_tool({
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    })
    text = (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="Write">{NS}<content>hello'
        f"{NS}</content>{NS}<filePath>/tmp/test.txt"
        f"{NS}</filePath>{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )
    calls, _ = _parse_tool_calls(text, minimax_m3, tools)
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {"file_path": "/tmp/test.txt", "content": "hello"}, args


def test_legacy_display_style_is_not_recovered_by_default():
    tools = write_tool({
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    })
    text = '[Tool call: Write]\n{"file_path": "/tmp/test.txt", "content": "hello"}'
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    assert calls == [], calls
    assert remaining == text, remaining


def main():
    assert TOOL_COMPAT_OVERLAY is False
    test_native_claude_style_write_call_with_camel_schema()
    test_native_claude_style_write_call_maps_to_snake_schema()
    test_legacy_display_style_is_not_recovered_by_default()
    print("PASS")


if __name__ == "__main__":
    main()
