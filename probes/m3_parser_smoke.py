#!/usr/bin/env python3
"""Smoke-check MiniMax reasoning/content marker parsing."""

import json
import os
import pathlib
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This probe intentionally exercises the legacy MiniMax/Codex compatibility
# overlay. Production defaults keep this overlay off for standard OpenAI tool
# clients; set it here before importing sharded_server so the old recovery paths
# remain regression-tested.
os.environ.setdefault("MLX_M3_TOOL_COMPAT_OVERLAY", "1")
os.environ.setdefault("MLX_M3_TOOL_SYSTEM_HINT", "1")
os.environ.setdefault("MLX_M3_INJECT_DATE_CONTEXT", "1")
os.environ.setdefault("MLX_M3_TOOL_LOOP_STEER_MAX_TOOL_ONLY_TURNS", "8")
os.environ.setdefault("MLX_M3_TOOL_LOOP_STEER_MAX_REPEATED_TOOL", "6")
os.environ.setdefault("MLX_M3_TOOL_LOOP_STEER_MAX_REPEATED_COMMANDS", "3")
os.environ.setdefault("MLX_M3_TOOL_LOOP_FORCE_FINAL_AFTER", "8")
os.environ.setdefault("MLX_M3_TOOL_INCOMPLETE_CALL_TOKEN_BUDGET", "8192")

from sharded_server import (
    _BATCH_PATH_ACTIVE,
    _FORCE_EOS,
    _add_date_system_context,
    _add_tool_system_hint_if_needed,
    _assistant_content_for_template,
    _anchor_command_working_directory,
    _repair_reversed_edit_arguments_after_failure,
    _bound_large_file_write_arguments,
    _date_context_for_session,
    _arm_rank0_semantic_eos,
    _filter_looping_control_tools,
    _file_write_chunk_hint,
    _file_write_payload_chars,
    _incomplete_tool_call_budget_reached,
    _looks_like_tool_compat_fallback_content,
    _remember_assistant_reasoning,
    _tool_request_fallback_content,
    _tool_retry_recovery_hint,
    _tool_loop_steering_diag,
    _usable_tool_turn,
    _parse_tool_calls,
    _sanitize_inbound_tool_call_content,
    _sanitize_inbound_message_content,
    _shell_create_file_payload_info,
    _synthesize_bounded_write_scaffold_text,
    _synthesize_write_command_tool_call,
    _tool_call_complete_for_stop,
    _tool_call_contains_complete_but_invalid,
    _validate_outgoing_tool_calls,
    split_stream_thinking_delta,
    split_thinking_text,
)
from model_gateway import _openai_response_has_usable_content


class FakeMiniMaxToolModule:
    tool_call_start = "]<]minimax[>[<tool_call>"
    tool_call_end = "]<]minimax[>[</tool_call>"

    @staticmethod
    def parse_tool_call(text, tools):
        # Minimal parser used only for this smoke test. The production path uses
        # mlx-vlm's tool parser first, then the ThunderMLX recovery helpers.
        import json
        import re

        ns = "]<]minimax[>["
        name_match = re.search(r'<invoke name="([^"]+)">', text)
        if not name_match:
            raise ValueError("missing invoke name")
        name = name_match.group(1)
        args = {}
        for param, value in re.findall(
            rf'{re.escape(ns)}<parameter name="([^"]+)">(.*?){re.escape(ns)}</[^>]+>',
            text,
            flags=re.DOTALL,
        ):
            args[param] = value
        return {"name": name, "arguments": json.dumps(args)}


def check_complete_analysis_channel():
    reasoning, content = split_thinking_text(
        "<|channel>analysis\ncheck cache<channel|>\nfinal answer",
        assume_in_thinking=False,
    )
    assert reasoning == "check cache", (reasoning, content)
    assert content == "final answer", (reasoning, content)


def check_stream_analysis_channel():
    in_thinking = False
    accumulated = ""
    at_start = True
    reasoning_out = []
    content_out = []
    for token in [
        "<|channel>analysis",
        "\ncheck cache",
        "<channel|>",
        "\nfinal answer",
    ]:
        accumulated += token
        (
            in_thinking,
            accumulated,
            at_start,
            delta_reasoning,
            delta_content,
        ) = split_stream_thinking_delta(
            accumulated,
            token,
            in_thinking,
            at_response_start=at_start,
        )
        if delta_reasoning:
            reasoning_out.append(delta_reasoning)
        if delta_content:
            content_out.append(delta_content)
    assert "check cache" in "".join(reasoning_out), reasoning_out
    assert "final answer" in "".join(content_out), content_out


def check_unknown_channel_does_not_buffer_forever():
    in_thinking = False
    accumulated = ""
    at_start = True
    content_out = []
    for token in [
        "<|channel>final",
        "<|message>",
        "ready",
    ]:
        accumulated += token
        (
            in_thinking,
            accumulated,
            at_start,
            delta_reasoning,
            delta_content,
        ) = split_stream_thinking_delta(
            accumulated,
            token,
            in_thinking,
            at_response_start=at_start,
        )
        assert not delta_reasoning, delta_reasoning
        if delta_content:
            content_out.append(delta_content)
    assert "ready" in "".join(content_out), content_out
    assert accumulated == "", accumulated


def check_malformed_positional_xml_tool_call_recovers():
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_search",
            "description": "Search project context.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>"
        f"{ns}<invoke execute_search>"
        f'<parameter name="query">obsidian vault location{ns}</query>'
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "execute_search", call
    assert '"query": "obsidian vault location"' in call["function"]["arguments"], call


def check_malformed_command_tag_tool_calls_recover():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    cases = [
        (
            f"{ns}<tool_call>"
            f'{ns}<invoke_command">'
            f'{ns}<cmd>ls -la "/Users/example/Projects/Mac Transfer/Visual Studio Code.app"{ns}</cmd>'
            f"{ns}<justification>Check VS Code app exists{ns}</justification>"
            f"{ns}</invoke>"
            f"{ns}</tool_call>"
        ),
        (
            f"{ns}<tool_call>"
            f"{ns}<invoke>"
            f'{ns}<cmd>APP="/Users/example/Projects/Mac Transfer/Visual Studio Code.app"; ls -la "$APP"{ns}</cmd>'
            f"{ns}<justification>Check VS Code app version{ns}</justification>"
            f"{ns}</invoke>"
            f"{ns}</tool_call>"
        ),
        (
            f"{ns}<tool_call>"
            f'{ns}<exec_command">'
            f'{ns}<cmd>ls -la "/Users/example/Projects/Mac Transfer/Visual Studio Code.app"{ns}</cmd>'
            f"{ns}<justification>Check VS Code app exists{ns}</justification>"
            f"{ns}</invoke>"
            f"{ns}</tool_call>"
        ),
    ]
    for text in cases:
        calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
        assert remaining == "", remaining
        assert len(calls) == 1, (text, calls)
        call = calls[0]
        assert call["function"]["name"] == "Bash", call
        assert "Visual Studio Code.app" in call["function"]["arguments"], call
        assert '"cmd":' in call["function"]["arguments"], call


def check_codex_pseudo_goal_call_recovers():
    tools = [{
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": "Create a persistent goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["objective"],
            },
        },
    }]
    text = (
        '<<< $goal = create_goal(objective: "Tell me what this app is and '
        'identify missed opportunities."); >>>'
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "create_goal", call
    assert '"objective":' in call["function"]["arguments"], call


def check_codex_pseudo_goal_before_malformed_exec_prefers_goal():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_goal",
                "parameters": {
                    "type": "object",
                    "properties": {"objective": {"type": "string"}},
                    "required": ["objective"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
    ]
    ns = "]<]minimax[>["
    text = (
        '<<< $goal = create_goal(objective: "Map the project structure."); >>>'
        f"{ns}<tool_call>"
        f'{ns}<exec_command">{ns}</invoke>'
        f"{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "create_goal", call
    assert '"objective": "Map the project structure."' in call["function"]["arguments"], call


def check_invoke_name_attr_drift_recovers():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>"
        f'{ns}<invoke_name="exec_command">'
        f"{ns}<cmd>ls /tmp | head{ns}</cmd>"
        f"{ns}</invoke>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "exec_command", call
    assert '"cmd": "ls /tmp | head"' in call["function"]["arguments"], call


def check_display_style_tool_call_recovers_and_strips():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        "Now I have a comprehensive understanding. "
        "Let me check a few more key files to round out the picture."
        f"{ns}\n[Tool call: exec]\n"
        "{\"cmd\":\"cat /tmp/example.swift | sed -n '900,1200p'\","
        '"timeout":10000}'
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == (
        "Now I have a comprehensive understanding. "
        "Let me check a few more key files to round out the picture."
    ), remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "exec_command", call
    assert '"cmd": "cat /tmp/example.swift | sed -n' in call["function"]["arguments"], call


def check_loose_segment_command_tool_call_recovers():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>\n"
        f"{ns}[]{ns}[python3 \u2013version]{ns}[]{ns}[Run project check]{ns}[]"
        f"\n{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "exec_command", call
    decoded = json.loads(call["function"]["arguments"])
    assert decoded["cmd"] == "python3 --version", decoded
    assert decoded["justification"] == "Run project check", decoded


def check_incomplete_loose_segment_command_is_not_emitted():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>\n"
        f"{ns}[]{ns}[python3 -c \u2019import]{ns}[]{ns}[Run project check]{ns}[]"
        f"\n{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert calls == [], calls
    assert remaining == "", remaining


def check_incomplete_native_write_is_never_emitted():
    tools = [{
        "type": "function",
        "function": {
            "name": "Write",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        "I will write it now."
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Write">'
        f'{ns}<parameter name="file_path">snake.html'
        f'{ns}</parameter>'
        f'{ns}<parameter name="content"><!doctype html><script>'
        "function step() {"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert calls == [], calls
    assert remaining == "I will write it now.", remaining
    recovery = _tool_retry_recovery_hint(
        text,
        FakeMiniMaxToolModule,
        tools,
    )
    assert "did not close before the generation budget ended" in recovery, recovery
    assert _file_write_payload_chars(text, tools) > 0, text
    assert not _incomplete_tool_call_budget_reached(
        8191,
        True,
        text,
        FakeMiniMaxToolModule,
    )
    assert _incomplete_tool_call_budget_reached(
        8192,
        True,
        text,
        FakeMiniMaxToolModule,
    )
    closed = text + f"{ns}</tool_call>"
    assert not _incomplete_tool_call_budget_reached(
        8192,
        True,
        closed,
        FakeMiniMaxToolModule,
    )


def check_large_write_chunk_hint_and_retry_feedback():
    tools = [{
        "type": "function",
        "function": {
            "name": "Write",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    }]
    hint = _file_write_chunk_hint(tools)
    assert "6000 characters" in hint, hint
    assert "small valid working scaffold" in hint, hint

    ns = "]<]minimax[>["
    malformed = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Write">'
        f'{ns}<parameter name="file_path">/tmp/snake.html'
        f'{ns}</parameter>'
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    recovery = _tool_retry_recovery_hint(
        malformed,
        FakeMiniMaxToolModule,
        tools,
    )
    assert "required argument(s) `content` were missing" in recovery, recovery
    assert "small working scaffold" in recovery, recovery

    edit_tools = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    }]
    incomplete_edit = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Edit">'
        f"{ns}<file_path>/tmp/snake.html{ns}</file_path>"
        f"{ns}<new_string>replacement without an old-string anchor"
    )
    edit_recovery = _tool_retry_recovery_hint(
        incomplete_edit,
        FakeMiniMaxToolModule,
        edit_tools,
    )
    assert "exactly one valid `Edit` call" in edit_recovery, edit_recovery
    assert "`old_string`" in edit_recovery, edit_recovery
    assert "`new_string`" in edit_recovery, edit_recovery
    assert "one focused replacement" in edit_recovery, edit_recovery

    import sharded_server as server

    original_chunk_limit = server.TOOL_WRITE_CHUNK_MAX_CHARS
    server.TOOL_WRITE_CHUNK_MAX_CHARS = 0
    try:
        unbounded_recovery = _tool_retry_recovery_hint(
            incomplete_edit,
            FakeMiniMaxToolModule,
            edit_tools,
        )
        assert "0 characters" not in unbounded_recovery, unbounded_recovery
        assert "one focused replacement" in unbounded_recovery, unbounded_recovery
    finally:
        server.TOOL_WRITE_CHUNK_MAX_CHARS = original_chunk_limit

    oversized_call = {
        "file_path": "/tmp/snake.html",
        "content": "x" * 7000,
    }
    bounded, original_chars = _bound_large_file_write_arguments(
        "Write",
        oversized_call,
    )
    assert original_chars == 7000, original_chars
    assert len(bounded["content"]) < 6000, bounded
    assert "THUNDERMLX_CONTINUE" in bounded["content"], bounded
    assert bounded["file_path"] == "/tmp/snake.html", bounded

    incomplete_large = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Write">'
        f"{ns}<file_path>/tmp/snake.html{ns}</file_path>"
        f"{ns}<content>"
        + ("x" * 7000)
    )
    scaffold_text = _synthesize_bounded_write_scaffold_text(
        incomplete_large,
        tools,
    )
    assert scaffold_text.startswith("[Tool call: Write]"), scaffold_text[:200]
    calls, remaining = _parse_tool_calls(
        scaffold_text,
        FakeMiniMaxToolModule,
        tools,
    )
    assert len(calls) == 1, calls
    assert remaining == "", remaining
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["file_path"] == "/tmp/snake.html", args
    assert "THUNDERMLX_CONTINUE" in args["content"], args
    assert len(args["content"]) < 6000, args

    shell_tools = [
        *tools,
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
    oversized_shell = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Bash">'
        f"{ns}<command>cat > /tmp/from-bash.html <<'EOF'\n"
        + ("x" * 7000)
    )
    shell_info = _shell_create_file_payload_info(
        oversized_shell,
        shell_tools,
    )
    assert shell_info, oversized_shell[:200]
    assert shell_info["path"] == "/tmp/from-bash.html", shell_info
    shell_scaffold = _synthesize_bounded_write_scaffold_text(
        oversized_shell,
        shell_tools,
    )
    calls, remaining = _parse_tool_calls(
        shell_scaffold,
        FakeMiniMaxToolModule,
        shell_tools,
    )
    assert len(calls) == 1, calls
    assert remaining == "", remaining
    assert calls[0]["function"]["name"] == "Write", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["file_path"] == "/tmp/from-bash.html", args
    assert "THUNDERMLX_CONTINUE" in args["content"], args

    ordinary_bash = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Bash">'
        f"{ns}<command>python3 -m pytest\n"
        + ("x" * 7000)
    )
    assert _shell_create_file_payload_info(ordinary_bash, shell_tools) is None
    assert not _synthesize_bounded_write_scaffold_text(
        ordinary_bash,
        shell_tools,
    )


def check_post_tool_action_promise_is_not_a_final_answer():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }]
    messages = [
        {"role": "user", "content": "Create the document."},
        {"role": "assistant", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command":"pwd"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp"},
    ]
    text = (
        "Now let me write a comprehensive Python script that builds the "
        "document. I'll create it in /tmp and run it to produce the final .docx."
    )
    assert not _usable_tool_turn(
        text,
        FakeMiniMaxToolModule,
        tools,
        messages,
        "enabled",
    )


def check_codex_tool_arg_schema_canonicalization():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "yield_time_ms": {"type": "integer"},
                },
                "required": ["cmd"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "invoke_command",
            "arguments": '{"command": "pwd", "justification": "check cwd"}',
        },
    }]
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    call = validated[0]
    assert call["function"]["name"] == "exec_command", call
    assert '"cmd": "pwd"' in call["function"]["arguments"], call
    assert "justification" not in call["function"]["arguments"], call


def check_command_workdir_drift_is_anchored_narrowly():
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    }]
    root = "/workspace/ThunderMLX/opencode-fixture"
    messages = [
        {
            "role": "system",
            "content": f"<env>\nWorking directory: {root}\n</env>",
        },
        {"role": "user", "content": "Run the unit tests."},
    ]
    repaired, changes = _anchor_command_working_directory(
        "bash",
        {
            "command": "python3 -m unittest",
            "workdir": "/tmp/other-checkout/opencode-fixture",
        },
        tools,
        messages,
    )
    assert repaired["workdir"] == root, repaired
    assert len(changes) == 1, changes

    subdir, changes = _anchor_command_working_directory(
        "bash",
        {"command": "pwd", "workdir": f"{root}/tests"},
        tools,
        messages,
    )
    assert subdir["workdir"] == f"{root}/tests", subdir
    assert not changes, changes

    external = "/tmp/opencode-fixture"
    explicit_messages = [
        messages[0],
        {"role": "user", "content": f"Run the check in {external}."},
    ]
    preserved, changes = _anchor_command_working_directory(
        "bash",
        {"command": "pwd", "workdir": external},
        tools,
        explicit_messages,
    )
    assert preserved["workdir"] == external, preserved
    assert not changes, changes


def check_reversed_edit_is_repaired_only_after_proven_failure():
    tools = [{
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string"},
                    "newString": {"type": "string"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    }]
    path = "/workspace/sample.py"
    current = "def main():\n    return 1"
    desired = "def helper():\n    return 2\n\n\ndef main():\n    return 1"
    read_id = "read-1"
    failed_id = "edit-1"
    messages = [
        {"role": "user", "content": "Add helper to sample.py."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": read_id,
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"filePath": path}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": read_id,
            "content": (
                f"<path>{path}</path>\n<content>\n"
                "1: def main():\n2:     return 1\n"
                "</content>"
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": failed_id,
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps({
                        "filePath": path,
                        "oldString": desired,
                        "newString": current,
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": failed_id,
            "content": (
                "Could not find oldString in the file. It must match "
                "exactly, including whitespace."
            ),
        },
    ]
    inverted = {
        "filePath": path,
        "oldString": desired,
        "newString": current,
    }
    repaired, evidence = _repair_reversed_edit_arguments_after_failure(
        "edit",
        inverted,
        messages,
    )
    assert evidence, (repaired, evidence)
    assert repaired["oldString"] == current, repaired
    assert repaired["newString"] == desired, repaired

    without_failure, evidence = _repair_reversed_edit_arguments_after_failure(
        "edit",
        inverted,
        messages[:-1],
    )
    assert without_failure == inverted, without_failure
    assert evidence is None, evidence

    valid = {
        "filePath": path,
        "oldString": current,
        "newString": desired,
    }
    preserved, evidence = _repair_reversed_edit_arguments_after_failure(
        "edit",
        valid,
        messages,
    )
    assert preserved == valid, preserved
    assert evidence is None, evidence


def check_exec_command_justification_is_filled_when_supported():
    import sharded_server as server

    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["cmd"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "arguments": {"cmd": "ls -la /tmp"},
        },
    }]
    previous = server.TOOL_SYNTH_JUSTIFICATION
    server.TOOL_SYNTH_JUSTIFICATION = True
    try:
        validated = _validate_outgoing_tool_calls(calls, tools)
    finally:
        server.TOOL_SYNTH_JUSTIFICATION = previous
    assert len(validated) == 1, validated
    call = validated[0]
    assert '"cmd": "ls -la /tmp"' in call["function"]["arguments"], call
    assert '"justification": "List files"' in call["function"]["arguments"], call


def check_post_tool_hint_tells_model_to_answer():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What app is this?"},
        {"role": "tool", "content": "Package.swift shows OpenScienceApp."},
    ]
    patched = _add_tool_system_hint_if_needed(
        messages,
        {"thinking_mode": "disabled"},
        tools,
    )
    assert patched[0]["role"] == "system", patched
    assert "provide the final answer now" in patched[0]["content"], patched[0]


def check_daily_date_context_injection():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What date is it?"},
    ]
    patched, injected = _add_date_system_context(messages)
    assert injected is True, patched
    assert patched[0]["role"] == "system", patched
    assert time.strftime("%Y-%m-%d") in patched[0]["content"], patched[0]
    assert "Treat this date as authoritative" in patched[0]["content"], patched[0]

    session_id = f"date-pin-{time.time_ns()}"
    first = _date_context_for_session(
        session_id,
        "Current date: 2030-01-02 (Wednesday).",
    )
    after_midnight = _date_context_for_session(
        session_id,
        "Current date: 2030-01-03 (Thursday).",
    )
    other_session = _date_context_for_session(
        session_id + "-new",
        "Current date: 2030-01-03 (Thursday).",
    )
    assert after_midnight == first, (first, after_midnight)
    assert other_session != first, (first, other_session)

    pinned_messages, injected = _add_date_system_context(
        messages,
        session_id=session_id,
    )
    assert injected is True, pinned_messages
    assert pinned_messages[0]["content"] == first, pinned_messages[0]

    supplied = [
        {
            "role": "system",
            "content": "Use this date/time context. Full current datetime: 2030-01-02.",
        },
        {"role": "user", "content": "What date is it?"},
    ]
    preserved, injected = _add_date_system_context(supplied)
    assert injected is False, preserved
    assert preserved == supplied, preserved


def check_thinking_tool_hint_covers_fresh_information():
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    messages = [{
        "role": "user",
        "content": "Who is playing in the World Cup today?",
    }]
    patched = _add_tool_system_hint_if_needed(
        messages,
        {"thinking_mode": "enabled"},
        tools,
    )
    assert patched[0]["role"] == "system", patched
    assert "current, external, or user-specific" in patched[0]["content"], patched[0]
    assert "otherwise answer directly" in patched[0]["content"], patched[0]


def check_soft_tool_loop_steering_preserves_tools():
    tools = [{
        "type": "function",
        "function": {
            "name": "update_plan",
            "parameters": {
                "type": "object",
                "properties": {"plan": {"type": "array"}},
                "required": ["plan"],
            },
        },
    }]
    normal = [
        {"role": "user", "content": "Inspect the project."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "update_plan", "arguments": {"plan": []}},
            }],
        },
        {"role": "tool", "content": "ok"},
    ]
    assert _tool_loop_steering_diag(normal, tools) is None

    looped = [{"role": "user", "content": "Inspect the project."}]
    for _ in range(6):
        looped.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "update_plan", "arguments": {"plan": []}},
            }],
        })
        looped.append({"role": "tool", "content": "ok"})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "repeated_tool" in diag["reasons"], diag
    patched = _add_tool_system_hint_if_needed(looped, {}, tools, tool_loop_diag=diag)
    assert patched[0]["role"] == "system", patched
    assert "Tools remain available" in patched[0]["content"], patched[0]
    assert "tool-call format" in patched[0]["content"], patched[0]


def _repeated_named_tool_messages(tool_name, count):
    messages = [{"role": "user", "content": "Research this and report back."}]
    for index in range(count):
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": {"query": f"topic {index}"},
                },
            }],
        })
        messages.append({"role": "tool", "content": f"result {index}"})
    return messages


def check_soft_tool_loop_hint_is_cache_stable():
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    hints = []
    for count in (6, 7):
        messages = _repeated_named_tool_messages("web_search", count)
        diag = _tool_loop_steering_diag(messages, tools)
        assert diag and "repeated_tool" in diag["reasons"], diag
        assert "force_final" not in diag["reasons"], diag
        patched = _add_tool_system_hint_if_needed(
            messages,
            {"thinking_mode": "enabled"},
            tools,
            tool_loop_diag=diag,
        )
        hints.append(patched[0]["content"])
    assert hints[0] == hints[1], hints


def check_targeted_repeated_tool_breaker_is_scoped():
    import sharded_server as server

    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("web_search", "read_file")
    ]
    previous_after = server.TOOL_LOOP_FORCE_FINAL_AFTER
    previous_count = server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT
    previous_names = server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES
    try:
        server.TOOL_LOOP_FORCE_FINAL_AFTER = 0
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT = 8
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES = {"web_search"}

        search_diag = _tool_loop_steering_diag(
            _repeated_named_tool_messages("web_search", 8),
            tools,
        )
        assert "repeated_tool_limit" in search_diag["reasons"], search_diag
        assert "force_final" in search_diag["reasons"], search_diag

        coding_diag = _tool_loop_steering_diag(
            _repeated_named_tool_messages("read_file", 8),
            tools,
        )
        assert coding_diag and "repeated_tool" in coding_diag["reasons"], coding_diag
        assert "force_final" not in coding_diag["reasons"], coding_diag
    finally:
        server.TOOL_LOOP_FORCE_FINAL_AFTER = previous_after
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_COUNT = previous_count
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_TOOL_NAMES = previous_names


def check_repeated_control_tool_filter_keeps_work_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "update_plan",
                "parameters": {
                    "type": "object",
                    "properties": {"plan": {"type": "array"}},
                    "required": ["plan"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
    ]
    looped = [{"role": "user", "content": "Inspect the project."}]
    for _ in range(6):
        looped.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "update_plan", "arguments": {"plan": []}},
            }],
        })
        looped.append({"role": "tool", "content": "ok"})
    diag = _tool_loop_steering_diag(looped, tools)
    filtered, names = _filter_looping_control_tools(tools, diag)
    assert names == ["update_plan"], names
    remaining = [item["function"]["name"] for item in filtered]
    assert remaining == ["exec_command"], remaining


def check_repeated_exec_command_keeps_tool_schema_stable():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        },
    ]
    looped = [{"role": "user", "content": "Create a small Python script."}]
    for _ in range(3):
        looped.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": {"cmd": "ls -la"},
                },
            }],
        })
        looped.append({"role": "tool", "content": "total 8\n-rw-r--r-- app.py"})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "repeated_command" in diag["reasons"], diag
    assert "force_final" not in diag["reasons"], diag
    filtered, names = _filter_looping_control_tools(tools, diag)
    assert names == [], names
    remaining = [item["function"]["name"] for item in filtered]
    assert remaining == ["exec_command", "apply_patch"], remaining


def check_repeated_exec_command_eventually_forces_final_answer():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        },
    ]
    looped = [{"role": "user", "content": "Create a small Python script."}]
    for _ in range(4):
        looped.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": {"cmd": "ls -la"},
                },
            }],
        })
        looped.append({"role": "tool", "content": "total 8\n-rw-r--r-- app.py"})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "repeated_command" in diag["reasons"], diag
    assert "force_final" not in diag["reasons"], diag
    patched = _add_tool_system_hint_if_needed(looped, {}, tools, tool_loop_diag=diag)
    assert patched[0]["role"] == "system", patched
    assert "Long agent-loop steering" in patched[0]["content"], patched[0]
    assert "tools are unavailable" not in patched[0]["content"], patched[0]


def check_repeated_apply_patch_keeps_tool_schema_stable():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
    ]
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-print('x')\n+print('hi')\n*** End Patch"
    looped = [{"role": "user", "content": "Patch the file."}]
    for _ in range(3):
        looped.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": {"patch": patch},
                },
            }],
        })
        looped.append({"role": "tool", "content": "patch failed: context not found"})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "repeated_command" in diag["reasons"], diag
    assert diag.get("repeated_tool") == "apply_patch", diag
    assert "force_final" not in diag["reasons"], diag
    filtered, names = _filter_looping_control_tools(tools, diag)
    assert names == [], names
    remaining = [item["function"]["name"] for item in filtered]
    assert remaining == ["apply_patch", "exec_command"], remaining


def check_long_tool_loop_gets_force_final_hint():
    tools = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
        },
    }]
    looped = [{"role": "user", "content": "Create a small Python script."}]
    for index in range(8):
        looped.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": {"patch": "*** Begin Patch\n*** End Patch"},
                },
            }],
        })
        looped.append({"role": "tool", "content": f"patch failed {index}"})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "force_final" in diag["reasons"], diag
    patched = _add_tool_system_hint_if_needed(looped, {}, None, tool_loop_diag=diag)
    assert patched[0]["role"] == "system", patched
    assert "stop calling tools for this single turn" in patched[0]["content"], patched[0]
    assert "Do not request another tool" in patched[0]["content"], patched[0]


def check_create_goal_objective_aliases():
    tools = [{
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": "Create a persistent goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["objective"],
            },
        },
    }]
    cases = [
        {"description": "Monitor the cluster until the agent run completes."},
        {"input": "Keep watching tool calls for stalls."},
        {"task": "Validate cache reuse and report failures."},
    ]
    for args in cases:
        calls = [{
            "type": "function",
            "function": {
                "name": "create_goal",
                "arguments": args,
            },
        }]
        validated = _validate_outgoing_tool_calls(calls, tools)
        assert len(validated) == 1, (args, validated)
        call = validated[0]
        assert call["function"]["name"] == "create_goal", call
        assert '"objective":' in call["function"]["arguments"], call


def check_update_plan_scalar_coerces_to_plan_item():
    tools = [{
        "type": "function",
        "function": {
            "name": "update_plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {"type": "string"},
                            },
                            "required": ["step", "status"],
                        },
                    }
                },
                "required": ["plan"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": {"plan": "Audit tool calls"},
        },
    }]
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {"plan": [{"step": "Audit tool calls", "status": "in_progress"}]}, args


def check_missing_required_tool_args_report_dropped():
    tools = [{
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": "Create a persistent goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "token_budget": {"type": "integer"},
                },
                "required": ["objective"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "create_goal",
            "arguments": {"token_budget": 1000},
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert validated == [], validated
    assert dropped == 1, dropped


def check_empty_required_tool_args_report_dropped():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "arguments": "{}",
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert validated == [], validated
    assert dropped == 1, dropped


def check_parameterless_and_optional_only_tool_args_are_accepted():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "skills_list",
                "description": "List installed skills.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_resources",
                "description": "List resources, optionally filtered.",
                "parameters": {
                    "type": "object",
                    "properties": {"server": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]
    calls = [
        {
            "type": "function",
            "function": {"name": "skills_list", "arguments": "{}"},
        },
        {
            "type": "function",
            "function": {"name": "list_resources", "arguments": {}},
        },
    ]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, dropped
    assert [call["function"]["name"] for call in validated] == [
        "skills_list",
        "list_resources",
    ], validated
    assert all(
        json.loads(call["function"]["arguments"]) == {}
        for call in validated
    ), validated


def check_malformed_apply_patch_payload_report_dropped():
    tools = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch.",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "arguments": {"input": "*** Begin Patch\n***"},
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert validated == [], validated
    assert dropped == 1, dropped

    valid_patch = (
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@\n"
        "-print('x')\n"
        "+print('hi')\n"
        "*** End Patch"
    )
    calls[0]["function"]["arguments"] = {"input": valid_patch}
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, dropped
    assert len(validated) == 1, validated


def check_tool_call_reasoning_recall_restores_model_context():
    session_id = "smoke-tool-reasoning-recall"
    tool_calls = [{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "exec_command",
            "arguments": json.dumps({"cmd": "printf 'hello\\n' > /tmp/hello.py"}),
        },
    }]
    raw_output = (
        "<mm:think>I need to create the file with a shell command.</mm:think>\n"
        "]<]minimax[>[<tool_call>\n"
        "]<]minimax[>[<invoke name=\"exec_command\">"
        "]<]minimax[>[<cmd>printf 'hello\\n' > /tmp/hello.py</cmd>"
        "]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    stored = _remember_assistant_reasoning(
        session_id,
        "",
        raw_output,
        thinking_mode="enabled",
        tool_calls=tool_calls,
    )
    assert stored, "expected reasoning recall to store tool-call reasoning"
    content = _assistant_content_for_template(
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        "",
        session_id=session_id,
    )
    assert content.startswith("<mm:think>"), content
    assert "I need to create the file" in content, content


def check_read_file_coerces_to_exec_command():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": ["cmd"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": {"path": "/tmp/example file.txt", "start": 3, "end": 8},
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, dropped
    assert len(validated) == 1, validated
    call = validated[0]
    assert call["function"]["name"] == "exec_command", call
    assert "sed -n '3,8p'" in call["function"]["arguments"], call
    assert "/tmp/example file.txt" in call["function"]["arguments"], call


def check_exec_stdin_aliases_to_write_stdin():
    tools = [{
        "type": "function",
        "function": {
            "name": "write_stdin",
            "description": "Send input to an existing session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "chars": {"type": "string"},
                },
                "required": ["session_id"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "exec_stdin",
            "arguments": {"session": 42, "stdin": "y\n"},
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, dropped
    assert len(validated) == 1, validated
    call = validated[0]
    assert call["function"]["name"] == "write_stdin", call
    assert '"session_id": 42' in call["function"]["arguments"], call
    assert '"chars": "y\\n"' in call["function"]["arguments"], call


def check_malformed_apply_patch_add_file_coerces_to_exec_write():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
        },
    ]
    calls = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            # Malformed: missing Begin/End envelope, but Add File intent intact.
            "arguments": {"input": "*** Add File: /tmp/notes/hello.txt\n+hello world"},
        },
    }]
    validated, dropped, dropped_names = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
        return_dropped_names=True,
    )
    assert dropped == 0, (dropped, dropped_names)
    assert len(validated) == 1, validated
    call = validated[0]
    assert call["function"]["name"] == "exec_command", call
    args = json.loads(call["function"]["arguments"])
    cmd = args.get("cmd") or args.get("command") or ""
    assert "mkdir -p /tmp/notes" in cmd, cmd
    assert "printf %s 'hello world' > /tmp/notes/hello.txt" in cmd, cmd
    # Update/Delete ops must never be reconstructed into shell writes.
    risky = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "arguments": {"input": "*** Update File: /tmp/notes/hello.txt\n+hello"},
        },
    }]
    validated, dropped, dropped_names = _validate_outgoing_tool_calls(
        risky,
        tools,
        return_dropped=True,
        return_dropped_names=True,
    )
    assert validated == [], validated
    assert dropped == 1, dropped


def check_stale_unavailable_tool_gets_compatibility_fallback():
    tools = [{
        "type": "function",
        "function": {
            "name": "tool_search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    calls = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "arguments": {"query": "latest MLX release notes"},
        },
    }]
    validated, dropped, dropped_names = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
        return_dropped_names=True,
    )
    assert validated == [], validated
    assert dropped == 1, dropped
    assert dropped_names == ["web_search"], dropped_names
    content = _tool_request_fallback_content(
        [{"role": "user", "content": "Search the web for MLX notes."}],
        dropped_tool_names=dropped_names,
        available_tool_names={"tool_search"},
    )
    assert "web_search" not in content, content
    assert "not available" not in content, content
    assert "malformed" not in content, content
    assert "tool schema" not in content, content
    assert "could not produce a valid tool call" in content, content
    assert _looks_like_tool_compat_fallback_content(content), content


def check_zcoder_codex_tool_argument_matrix():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_type": {"type": "string"},
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "justification": {"type": "string"},
                    },
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "WebFetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "parameters": {
                    "type": "object",
                    "properties": {"todos": {"type": "array"}},
                    "required": ["todos"],
                },
            },
        },
    ]
    cases = [
        ("Agent", {"type": "general-purpose", "task": "summarize the repo"}),
        ("Bash", {"command": "pwd"}),
        ("Read", {"path": "/tmp/example.swift", "offset": 10, "limit": 40}),
        (
            "Edit",
            {
                "path": "/tmp/example.swift",
                "old_text": "let a = 1",
                "new_text": "let a = 2",
            },
        ),
        ("Write", {"path": "/tmp/new.txt", "text": "hello"}),
        ("WebFetch", {"input": "https://example.com", "query": "summarize"}),
        ("TodoWrite", {"items": [{"content": "inspect", "status": "pending"}]}),
    ]
    for name, args in cases:
        validated, dropped = _validate_outgoing_tool_calls(
            [{"type": "function", "function": {"name": name, "arguments": args}}],
            tools,
            return_dropped=True,
        )
        assert dropped == 0, (name, args, dropped, validated)
        assert len(validated) == 1, (name, args, validated)
        call = validated[0]
        assert call["function"]["name"] == name, call
        decoded = json.loads(call["function"]["arguments"])
        required = {
            "Agent": ["prompt"],
            "Bash": ["cmd"],
            "Read": ["file_path"],
            "Edit": ["file_path", "old_string", "new_string"],
            "Write": ["file_path", "content"],
            "WebFetch": ["url"],
            "TodoWrite": ["todos"],
        }[name]
        for key in required:
            assert key in decoded and decoded[key] not in (None, ""), (name, decoded)


def check_empty_tool_markers_get_specific_fallback():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = f"{ns}<tool_call>\n{ns}<tool_call>\n{ns}<tool_call>"
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert calls == [], calls
    assert remaining == "", remaining
    content = _tool_request_fallback_content(
        [{"role": "user", "content": "List files."}],
        empty_tool_markers=True,
        thinking_mode="enabled",
    )
    assert "empty tool-call markers" not in content, content
    assert "retry" not in content.lower(), content
    assert "could not produce a valid tool call" in content, content
    assert "not executed" in content, content
    assert _looks_like_tool_compat_fallback_content(content), content


def check_tool_fallback_content_does_not_poison_next_turn():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    fallback = _tool_request_fallback_content(
        [{"role": "user", "content": "List files."}],
        empty_tool_markers=True,
        thinking_mode="enabled",
    )
    assert _sanitize_inbound_message_content("assistant", fallback) == ""
    looped = [{"role": "user", "content": "Inspect the project."}]
    for _ in range(2):
        looped.append({"role": "assistant", "content": fallback})
    diag = _tool_loop_steering_diag(looped, tools)
    assert diag and "tool_fallback_loop" in diag["reasons"], diag
    assert "force_final" in diag["reasons"], diag


def check_gateway_treats_tool_fallback_as_unusable():
    legacy = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": (
                    "I could not complete that tool step from the available "
                    "evidence. Here is the best answer from the context "
                    "already gathered."
                ),
            }
        }]
    }
    assert not _openai_response_has_usable_content(legacy), legacy
    current = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": _tool_request_fallback_content(
                    [{"role": "user", "content": "List files."}],
                    empty_tool_markers=True,
                ),
            }
        }]
    }
    assert not _openai_response_has_usable_content(current), current


def check_repeated_long_user_tool_prompt_forces_final():
    tools = [{
        "type": "function",
        "function": {
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    repeated = "Inspect this project and use tools as needed. " * 130
    messages = []
    for _ in range(3):
        messages.append({"role": "user", "content": repeated})
        messages.append({
            "role": "assistant",
            "content": "The previous tool action was incomplete and was not executed.",
        })
    diag = _tool_loop_steering_diag(messages, tools)
    assert diag and "repeated_user_tool_prompt" in diag["reasons"], diag
    assert "force_final" in diag["reasons"], diag


def check_inbound_tool_call_content_is_not_model_facing():
    message = {
        "role": "assistant",
        "content": "Calling exec_command now.",
        "tool_calls": [{"type": "function"}],
    }
    assert _sanitize_inbound_tool_call_content(message, message["content"]) == ""


def check_tool_stop_on_valid_or_complete_invalid_tool_call():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }]
    ns = "]<]minimax[>["
    invalid = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Bash">'
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    valid = (
        f"{ns}<tool_call>"
        f'{ns}<Bash>'
        f"{ns}<cmd>pwd{ns}</cmd>"
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    assert _tool_call_complete_for_stop(invalid, FakeMiniMaxToolModule, tools)
    assert _tool_call_contains_complete_but_invalid(invalid, FakeMiniMaxToolModule, tools)
    assert _tool_call_complete_for_stop(valid, FakeMiniMaxToolModule, tools)
    # An UNCLOSED block must never trigger a decode stop: judging a partial
    # emission truncates the model mid-call (`cat >` losing its filename, a
    # patch cut at 31 tokens) and manufactures malformed tool calls.
    for partial in (
        f"{ns}<tool_call>{ns}<invoke name=\"Bash\">{ns}<cmd>cat >",
        f"{ns}<tool_call>{ns}<invoke name=\"Bash\">",
        f"{ns}<tool_call>",
    ):
        assert not _tool_call_complete_for_stop(partial, FakeMiniMaxToolModule, tools), partial
        assert not _tool_call_contains_complete_but_invalid(partial, FakeMiniMaxToolModule, tools), partial


def check_invalid_apply_patch_stops_decode_without_emitting_tool():
    tools = [{
        "type": "function",
        "function": {
            "name": "apply_patch",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="apply_patch">'
        f"{ns}<patch>*** Begin Patch\n***{ns}</patch>"
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    calls, _ = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert calls, calls
    assert _validate_outgoing_tool_calls(calls, tools) == []
    assert _tool_call_complete_for_stop(text, FakeMiniMaxToolModule, tools)
    assert _tool_call_contains_complete_but_invalid(text, FakeMiniMaxToolModule, tools)


def check_malformed_apply_patch_simple_write_synthesizes_exec():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "justification": {"type": "string"},
                    },
                    "required": ["cmd"],
                },
            },
        },
    ]
    processed_messages = [{
        "role": "user",
        "content": (
            "Create a simple text file on the Desktop at "
            "/Users/example/Desktop/test.txt containing exactly hello, then answer done."
        ),
    }]
    call = _synthesize_write_command_tool_call(
        processed_messages,
        tools,
        dropped_tool_names=["apply_patch"],
    )
    assert call is not None, call
    assert call["function"]["name"] == "exec_command", call
    args = json.loads(call["function"]["arguments"])
    assert "printf %s hello" in args["cmd"], args
    assert "/Users/example/Desktop/test.txt" in args["cmd"], args


def check_semantic_decode_stop_is_rank0_owned():
    original = dict(_FORCE_EOS)
    original_batch = dict(_BATCH_PATH_ACTIVE)
    try:
        _FORCE_EOS["active"] = False
        _FORCE_EOS["eos_id"] = 123
        _BATCH_PATH_ACTIVE["value"] = True
        assert not _arm_rank0_semantic_eos(1, "test_guard", 17)
        assert not _FORCE_EOS["active"]
        assert _arm_rank0_semantic_eos(0, "test_guard", 17)
        assert _FORCE_EOS["active"]
        _FORCE_EOS["active"] = False
        _BATCH_PATH_ACTIVE["value"] = False
        assert not _arm_rank0_semantic_eos(0, "test_guard", 18)
        assert not _FORCE_EOS["active"]
    finally:
        _FORCE_EOS.clear()
        _FORCE_EOS.update(original)
        _BATCH_PATH_ACTIVE.clear()
        _BATCH_PATH_ACTIVE.update(original_batch)


def main():
    check_complete_analysis_channel()
    check_stream_analysis_channel()
    check_unknown_channel_does_not_buffer_forever()
    check_malformed_positional_xml_tool_call_recovers()
    check_malformed_command_tag_tool_calls_recover()
    check_codex_pseudo_goal_call_recovers()
    check_codex_pseudo_goal_before_malformed_exec_prefers_goal()
    check_invoke_name_attr_drift_recovers()
    check_display_style_tool_call_recovers_and_strips()
    check_loose_segment_command_tool_call_recovers()
    check_incomplete_loose_segment_command_is_not_emitted()
    check_incomplete_native_write_is_never_emitted()
    check_large_write_chunk_hint_and_retry_feedback()
    check_post_tool_action_promise_is_not_a_final_answer()
    check_codex_tool_arg_schema_canonicalization()
    check_command_workdir_drift_is_anchored_narrowly()
    check_reversed_edit_is_repaired_only_after_proven_failure()
    check_exec_command_justification_is_filled_when_supported()
    check_post_tool_hint_tells_model_to_answer()
    check_daily_date_context_injection()
    check_thinking_tool_hint_covers_fresh_information()
    check_soft_tool_loop_steering_preserves_tools()
    check_soft_tool_loop_hint_is_cache_stable()
    check_targeted_repeated_tool_breaker_is_scoped()
    check_repeated_control_tool_filter_keeps_work_tools()
    check_repeated_exec_command_keeps_tool_schema_stable()
    check_repeated_exec_command_eventually_forces_final_answer()
    check_repeated_apply_patch_keeps_tool_schema_stable()
    check_long_tool_loop_gets_force_final_hint()
    check_create_goal_objective_aliases()
    check_update_plan_scalar_coerces_to_plan_item()
    check_missing_required_tool_args_report_dropped()
    check_empty_required_tool_args_report_dropped()
    check_parameterless_and_optional_only_tool_args_are_accepted()
    check_malformed_apply_patch_payload_report_dropped()
    check_tool_call_reasoning_recall_restores_model_context()
    check_read_file_coerces_to_exec_command()
    check_exec_stdin_aliases_to_write_stdin()
    check_malformed_apply_patch_add_file_coerces_to_exec_write()
    check_stale_unavailable_tool_gets_compatibility_fallback()
    check_zcoder_codex_tool_argument_matrix()
    check_empty_tool_markers_get_specific_fallback()
    check_tool_fallback_content_does_not_poison_next_turn()
    check_gateway_treats_tool_fallback_as_unusable()
    check_repeated_long_user_tool_prompt_forces_final()
    check_inbound_tool_call_content_is_not_model_facing()
    check_tool_stop_on_valid_or_complete_invalid_tool_call()
    check_invalid_apply_patch_stops_decode_without_emitting_tool()
    check_malformed_apply_patch_simple_write_synthesizes_exec()
    check_semantic_decode_stop_is_rank0_owned()
    print("PASS")


if __name__ == "__main__":
    main()
