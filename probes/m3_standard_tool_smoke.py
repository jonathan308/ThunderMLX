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
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLX_M3_TOOL_COMPAT_OVERLAY", "0")
os.environ.setdefault("MLX_M3_TOOL_SYSTEM_HINT", "0")
os.environ.setdefault("MLX_M3_TOOL_STREAM_BUFFER_ALL", "0")
os.environ.setdefault("MLX_M3_TOOL_STREAM_CONTENT", "1")
os.environ.setdefault("MLX_M3_TOOL_THINKING_RUNAWAY_TOKEN_BUDGET", "0")
os.environ.setdefault("MLX_M3_TOOL_NO_CALL_TOKEN_BUDGET", "0")
os.environ.setdefault("MLX_M3_TOOL_ACTION_NO_CALL_TOKEN_BUDGET", "0")

import mlx_vlm.tool_parsers.minimax_m3 as minimax_m3
import sharded_server as server

from sharded_server import (
    NATIVE_TOOL_ACTION_RETRY_ATTEMPTS,
    NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS,
    TOOL_ACTION_NO_CALL_TOKEN_BUDGET,
    TOOL_COMPAT_OVERLAY,
    TOOL_NO_CALL_TOKEN_BUDGET,
    TOOL_STREAM_BUFFER_ALL,
    TOOL_STREAM_CONTENT,
    TOOL_SYSTEM_HINT_ENABLED,
    TOOL_THINKING_RUNAWAY_TOKEN_BUDGET,
    _parse_tool_calls,
    _native_tool_retry_ram_reset_reason,
    _tool_text_requests_action,
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


def todo_tool():
    return [{
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "Create and update the complete task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                    ],
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                            },
                            "required": ["content", "status", "priority"],
                        },
                    },
                },
                "required": ["todos"],
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


def test_large_native_write_roundtrips_without_bounding():
    """A complete native Write is atomic and must remain byte-for-byte intact."""
    tools = write_tool({
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    })
    content = "0123456789abcdef" * 1400
    assert len(content) > 20_000
    text = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Write">'
        f"{NS}<file_path>/tmp/large-native-write.html{NS}</file_path>"
        f"{NS}<content>{content}{NS}</content>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert remaining == "", remaining
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["file_path"] == "/tmp/large-native-write.html", args
    assert args["content"] == content


def test_native_paths_and_duplicate_calls_are_preserved():
    """Native mode must not anchor paths or silently suppress client actions."""
    tools = write_tool({
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    })
    arguments = {
        "file_path": "/Users/example/project/output.txt",
        "content": "same payload",
    }
    calls = [
        {
            "type": "function",
            "id": f"call-{index}",
            "function": {
                "name": "Write",
                "arguments": json.dumps(arguments),
            },
        }
        for index in range(2)
    ]
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 2, validated
    assert [
        json.loads(call["function"]["arguments"])
        for call in validated
    ] == [arguments, arguments]


def test_native_responses_style_to_attribute_recovers_exact_schema_name():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }]
    text = (
        f"{NS}<tool_call>"
        f'{NS}<invoke to="bash">'
        f"{NS}<command>ls -la /tmp{NS}</command>"
        f"{NS}<description>Inspect output{NS}</description>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert remaining == "", remaining
    assert len(validated) == 1, validated
    fn = validated[0]["function"]
    assert fn["name"] == "Bash", fn
    assert json.loads(fn["arguments"]) == {
        "command": "ls -la /tmp",
        "description": "Inspect output",
    }, fn


def test_native_action_reasoning_only_turn_gets_one_native_retry():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }]
    recovered = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Bash">'
        f"{NS}<command>pwd{NS}</command>"
        f"{NS}<description>Inspect working directory{NS}</description>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    messages = [{
        "role": "user",
        "content": "Use Bash to inspect the working directory.",
    }]
    with (
        patch.object(server, "_bcast") as broadcast,
        patch.object(server, "_clear_stop_request"),
        patch.object(server, "_clear_prefill_stop_file"),
        patch.object(server, "run_generation", return_value=recovered) as generate,
    ):
        output = server._ensure_usable_tool_turn(
            object(),
            object(),
            0,
            full_output="I should inspect the working directory first.",
            rank_request={"tool_choice": "auto"},
            prompt="prompt",
            max_tokens=512,
            thinking_mode="enabled",
            gen_params={"temperature": 0.0},
            image_path=None,
            token_ids=[1, 2, 3],
            session_id="native-retry-smoke",
            session_source="test",
            tool_module=minimax_m3,
            tools=tools,
            processed_messages=messages,
            req_id="native-retry-smoke",
            stream=True,
            action_tool_task=True,
        )
    assert output == recovered, output
    assert generate.call_count == 1, generate.call_count
    assert broadcast.call_count == 1, broadcast.call_count
    retry_request = broadcast.call_args.args[0]
    assert retry_request["thinking_mode"] == "enabled", retry_request
    assert retry_request["no_call_token_budget"] > 0, retry_request


def test_native_incomplete_call_after_tool_result_gets_one_retry():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }]
    incomplete = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Bash">'
        f"{NS}<command>pwd"
    )
    recovered = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Bash">'
        f"{NS}<command>pwd{NS}</command>"
        f"{NS}<description>Inspect working directory{NS}</description>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-plan",
                "type": "function",
                "function": {
                    "name": "TodoWrite",
                    "arguments": json.dumps({"todos": []}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-plan",
            "content": "[]",
        },
    ]
    with (
        patch.object(server, "_bcast") as broadcast,
        patch.object(server, "_clear_stop_request"),
        patch.object(server, "_clear_prefill_stop_file"),
        patch.object(server, "_tool_retry_recovery_hint", return_value=None),
        patch.object(server, "run_generation", return_value=recovered) as generate,
    ):
        output = server._ensure_usable_tool_turn(
            object(),
            object(),
            0,
            full_output=incomplete,
            rank_request={"tool_choice": "auto"},
            prompt="prompt",
            max_tokens=512,
            thinking_mode="disabled",
            gen_params={"temperature": 0.0},
            image_path=None,
            token_ids=[1, 2, 3],
            session_id="native-post-tool-retry-smoke",
            session_source="test",
            tool_module=minimax_m3,
            tools=tools,
            processed_messages=messages,
            req_id="native-post-tool-retry-smoke",
            stream=True,
            action_tool_task=False,
        )
    assert output == recovered, output
    assert generate.call_count == 1, generate.call_count
    assert broadcast.call_count == 1, broadcast.call_count


def test_corrupt_long_native_retry_releases_only_live_ram_kv():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }]
    recovered = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Bash">'
        f"{NS}<command>pwd{NS}</command>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    malformed = (
        f"{NS}<tool_call>{NS}<invoke name=\"Bash\">"
        f"{NS}<command>pwd" + ("\x00" * 32)
    )
    assert _native_tool_retry_ram_reset_reason(
        malformed,
        NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS,
    ) == "corrupt_control_bytes"
    with (
        patch.object(server, "_bcast") as broadcast,
        patch.object(server, "_clear_stop_request"),
        patch.object(server, "_clear_prefill_stop_file"),
        patch.object(server, "_tool_retry_recovery_hint", return_value=None),
        patch.object(server, "_reset_prompt_cache_on_all_ranks") as reset,
        patch.object(server, "run_generation", return_value=recovered) as generate,
    ):
        output = server._ensure_usable_tool_turn(
            object(),
            object(),
            0,
            full_output=malformed,
            rank_request={"tool_choice": "auto"},
            prompt="prompt",
            max_tokens=512,
            thinking_mode="disabled",
            gen_params={"temperature": 0.0},
            image_path=None,
            token_ids=list(range(NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS)),
            session_id="native-corrupt-retry-smoke",
            session_source="test",
            tool_module=minimax_m3,
            tools=tools,
            processed_messages=[{
                "role": "user",
                "content": "Use Bash to inspect the working directory.",
            }],
            req_id="native-corrupt-retry-smoke",
            stream=True,
            action_tool_task=True,
        )
    assert output == recovered, output
    assert generate.call_count == 1, generate.call_count
    assert broadcast.call_count == 1, broadcast.call_count
    reset.assert_called_once_with(
        0,
        reason="native tool retry RAM reset:corrupt_control_bytes",
        clear_memory=True,
        clear_manifest=False,
        clear_resident=False,
    )


def test_tool_decode_reuse_override_is_request_scoped():
    class FakeLanguage:
        _MSA_DECODE_TOPK_REUSE_TOKENS = 48

        @classmethod
        def set_decode_topk_reuse_tokens(cls, value):
            cls._MSA_DECODE_TOPK_REUSE_TOKENS = int(value)
            return cls._MSA_DECODE_TOPK_REUSE_TOKENS

    with patch.object(
        server,
        "_decode_topk_language_module",
        return_value=FakeLanguage,
    ):
        state = server._begin_request_decode_topk_reuse(
            [{"type": "function"}],
            0,
        )
        try:
            assert state is not None
            assert FakeLanguage._MSA_DECODE_TOPK_REUSE_TOKENS == 0
        finally:
            server._restore_request_decode_topk_reuse(state, 0)
        assert FakeLanguage._MSA_DECODE_TOPK_REUSE_TOKENS == 48

        state = server._begin_request_decode_topk_reuse([], 0)
        assert state is None
        assert FakeLanguage._MSA_DECODE_TOPK_REUSE_TOKENS == 48


def test_legacy_display_style_is_not_recovered_by_default():
    tools = write_tool({
        "file_path": {"type": "string"},
        "content": {"type": "string"},
    })
    text = '[Tool call: Write]\n{"file_path": "/tmp/test.txt", "content": "hello"}'
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    assert calls == [], calls
    assert remaining == text, remaining


def test_closed_native_write_survives_unfinished_followup():
    tools = [
        *write_tool({
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        }),
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    content = "<!doctype html><html><body>ready</body></html>"
    text = (
        "I will create the file."
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Write">'
        f"{NS}<file_path>/tmp/dino.html{NS}</file_path>"
        f"{NS}<content>{content}{NS}</content>"
        f"{NS}</invoke>"
        f"{NS}</tool_call>"
        "I will verify it."
        f'{NS}<tool_call> Bash {{"command":"ls -la /tmp/dino.html"}}'
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    assert validated[0]["function"]["name"] == "Write", validated
    assert json.loads(validated[0]["function"]["arguments"]) == {
        "file_path": "/tmp/dino.html",
        "content": content,
    }
    assert remaining == "I will create the file.", remaining


def test_closed_inner_invoke_does_not_salvage_unclosed_outer_write():
    """An inner Bash closer cannot make an abandoned Write atomic.

    This is the compact form of a live ZCode failure where MiniMax emitted an
    18.7 KB HTML game in an unclosed Write, then a closed Bash invoke in a
    second unclosed tool block. The old fallback returned only the HTML title
    as Write.content and the client faithfully replaced the file with it.
    """
    tools = [
        *write_tool({
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        }),
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    text = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Write">'
        "<file>/tmp/goal_game.html</file>"
        "<content><!doctype html><html><head>"
        "<title>Star Catcher - A Tiny Arcade Game</title>"
        "</head><body><script>requestAnimationFrame(loop)</script></body></html>"
        "\nNow I will verify it.</mm:think>"
        f"{NS}<tool_call>"
        f'{NS}<invoke name="Bash">'
        f"{NS}<command>wc -c /tmp/goal_game.html{NS}</command>"
        f"{NS}<description>Verify the file{NS}</description>"
        f"{NS}</invoke>"
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, tools)
    assert calls == [], calls
    assert remaining == "", remaining


def test_incomplete_namespaced_todo_items_are_not_invented():
    """Do not manufacture fields from the exact malformed live todo shape."""
    text = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="TodoWrite">'
        f"{NS}<todos>{NS}<item>"
        f"{NS}<content>Build the game</content>\n"
        "<status>in_progress</status>"
        f"{NS}<ACTIVEForm>Building the game</ACTIVEForm>"
        f"{NS}</item>{NS}<item>"
        f"{NS}<content>Verify the game</content>\n"
        f"<status>pending{NS}</status>"
        f"{NS}<ACTIVEForm>Verifying the game</ACTIVEForm>"
        f"{NS}</item>{NS}<item>"
        f"{NS}<content>Complete the game</content>\n"
        f"<status>pending{NS} status>"
        f"{NS}<ACTIVEForm>Completing the game</ACTIVEForm>"
        f"{NS}</item>{NS}</todos>{NS}</invoke>{NS}</tool_call>"
    )
    calls, _ = _parse_tool_calls(text, minimax_m3, todo_tool())
    validated = _validate_outgoing_tool_calls(calls, todo_tool())
    assert validated == [], validated


def test_wrapperless_complete_todo_retry_recovers_exact_schema():
    """Recover only complete repeated items from the live ZCode retry form."""
    text = (
        f"{NS}<tool_call>"
        f'{NS}<invoke name="TodoWrite">'
        f"{NS}<content>Build the game{NS}</content>"
        f"{NS}<status>in_progress{NS}</status>"
        f"{NS}<priority>high{NS}</priority>{NS}</item>"
        f"{NS}<content>Verify the game{NS}</content>"
        f"{NS}<status>pending{NS}</status>"
        f"{NS}<priority>medium{NS}</priority>{NS}</item>"
        f"{NS}</invoke>{NS}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, minimax_m3, todo_tool())
    validated = _validate_outgoing_tool_calls(calls, todo_tool())
    assert remaining == "", remaining
    assert len(validated) == 1, validated
    assert json.loads(validated[0]["function"]["arguments"]) == {
        "todos": [
            {
                "content": "Build the game",
                "status": "in_progress",
                "priority": "high",
            },
            {
                "content": "Verify the game",
                "status": "pending",
                "priority": "medium",
            },
        ],
    }


def main():
    assert TOOL_COMPAT_OVERLAY is False
    assert TOOL_SYSTEM_HINT_ENABLED is False
    assert TOOL_STREAM_BUFFER_ALL is False
    assert TOOL_STREAM_CONTENT is True
    assert TOOL_THINKING_RUNAWAY_TOKEN_BUDGET == 0
    assert TOOL_NO_CALL_TOKEN_BUDGET == 0
    assert TOOL_ACTION_NO_CALL_TOKEN_BUDGET == 0
    assert NATIVE_TOOL_ACTION_RETRY_ATTEMPTS == 1
    assert NATIVE_TOOL_ACTION_RETRY_RAM_RESET_TOKENS == 65536
    assert _tool_text_requests_action("Send the email using the terminal tool.")
    assert _tool_text_requests_action(
        "Set a goal to create a fun small HTML game in this workspace."
    )
    test_native_claude_style_write_call_with_camel_schema()
    test_native_claude_style_write_call_maps_to_snake_schema()
    test_large_native_write_roundtrips_without_bounding()
    test_native_paths_and_duplicate_calls_are_preserved()
    test_native_responses_style_to_attribute_recovers_exact_schema_name()
    test_native_action_reasoning_only_turn_gets_one_native_retry()
    test_native_incomplete_call_after_tool_result_gets_one_retry()
    test_corrupt_long_native_retry_releases_only_live_ram_kv()
    test_tool_decode_reuse_override_is_request_scoped()
    test_legacy_display_style_is_not_recovered_by_default()
    test_closed_native_write_survives_unfinished_followup()
    test_closed_inner_invoke_does_not_salvage_unclosed_outer_write()
    test_incomplete_namespaced_todo_items_are_not_invented()
    test_wrapperless_complete_todo_retry_recovers_exact_schema()
    print("PASS")


if __name__ == "__main__":
    main()
