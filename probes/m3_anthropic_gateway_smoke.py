#!/usr/bin/env python3
"""Smoke-check the Anthropic Messages gateway translator.

This does not require the model to be loaded. It verifies the Claude Code-facing
schema bridge: Anthropic tools/history become OpenAI tools/messages, and M3
OpenAI tool_calls become Anthropic tool_use blocks.
"""

from __future__ import annotations

import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_gateway import (  # noqa: E402
    _anthropic_bash_command_mutates,
    _anthropic_coding_continuation_message,
    _anthropic_declared_working_directory,
    _anthropic_exact_reply_after_verified_tool,
    _anthropic_has_tool_result,
    _anthropic_pending_mutation,
    _anthropic_write_fallback_message,
    _prune_openai_tools_for_anthropic_action,
    _repair_anthropic_tool_call_paths,
    _required_tool_retry_payload,
    anthropic_to_openai_payload,
    openai_to_anthropic_message,
)


def test_request_translation():
    payload = {
        "model": "sonnet",
        "system": [{"type": "text", "text": "Use tools when needed."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "create a file"}]},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Write",
                    "input": {"file_path": "/tmp/a.txt", "content": "hello"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "ok"}],
                }],
            },
        ],
        "tools": [{
            "name": "Write",
            "description": "Write a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        }],
        "tool_choice": {"type": "tool", "name": "Write"},
        "max_tokens": 128,
    }
    out = anthropic_to_openai_payload(payload)
    assert out["model"] == "Minimax-M3", out
    assert out["messages"][0]["role"] == "system"
    assert "make_file" in out["messages"][0]["content"]
    assert "Bash" in out["messages"][0]["content"]
    assert out["messages"][1] == {"role": "system", "content": "Use tools when needed."}
    assert out["messages"][3]["tool_calls"][0]["function"]["name"] == "make_file"
    args = json.loads(out["messages"][3]["tool_calls"][0]["function"]["arguments"])
    assert args == {"filename": "/tmp/a.txt", "content": "hello"}, args
    assert out["messages"][4] == {"role": "user", "content": "Tool result toolu_1:\nok"}
    assert out["tools"][0]["function"]["name"] == "make_file"
    assert out["tools"][0]["function"]["parameters"]["required"] == ["filename", "content"]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "make_file"}}
    assert out["_anthropic_tool_aliases"] == {
        "make_file": {
            "name": "Write",
            "arg_map": {"filename": "file_path", "content": "content"},
        }
    }


def test_response_translation():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": json.dumps({
                            "file_path": "/tmp/a.txt",
                            "content": "hello",
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_to_anthropic_message(response, "Minimax-M3")
    assert out["stop_reason"] == "tool_use", out
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}, out
    assert out["content"] == [{
        "type": "tool_use",
        "id": "call_1",
        "name": "Write",
        "input": {"file_path": "/tmp/a.txt", "content": "hello"},
    }], out


def test_response_translation_with_alias():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "make_file",
                        "arguments": json.dumps({
                            "filename": "/tmp/b.txt",
                            "content": "world",
                        }),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 6},
    }
    out = openai_to_anthropic_message(
        response,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "file_path", "content": "content"}}},
    )
    assert out["stop_reason"] == "tool_use", out
    assert out["content"] == [{
        "type": "tool_use",
        "id": "call_2",
        "name": "Write",
        "input": {"file_path": "/tmp/b.txt", "content": "world"},
    }], out


def test_write_fallback_extraction():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": "In the current directory, create result.txt containing exactly thundermlx-tool-ok, then reply with exactly done.",
        }],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "file_path", "content": "content"}}},
    )
    assert out is not None
    tool = out["content"][0]
    assert tool["name"] == "Write"
    assert tool["input"] == {"file_path": "result.txt", "content": "thundermlx-tool-ok"}, tool


def test_write_fallback_prefers_bash_when_available():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": "Create result.txt containing exactly thundermlx-tool-ok, then reply done.",
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
            {"name": "Write", "input_schema": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}}},
        ],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}},
    )
    assert out is not None
    tool = out["content"][0]
    assert tool["name"] == "Bash"
    assert "Path('result.txt').write_text('thundermlx-tool-ok')" in tool["input"]["command"], tool


def test_write_fallback_works_with_bash_without_write_alias():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": "Create result.txt containing exactly thundermlx-tool-ok, then reply done.",
        }],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_write_fallback_message(payload, "Minimax-M3", {}, simple_only=False)
    assert out is not None
    tool = out["content"][0]
    assert tool["name"] == "Bash"
    assert "Path('result.txt').write_text('thundermlx-tool-ok')" in tool["input"]["command"], tool


def test_write_fallback_ignores_system_reminder_paths():
    payload = {
        "model": "Minimax-M3",
        "messages": [
            {
                "role": "user",
                "content": "In the current directory, create result.txt containing exactly thundermlx-tool-ok, then reply done.",
            },
            {
                "role": "user",
                "content": "<system-reminder>Do not write to /system-reminder></system-reminder>",
            },
        ],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}},
    )
    assert out is not None
    tool = out["content"][0]
    assert "Path('result.txt')" in tool["input"]["command"], tool
    assert "/system-reminder" not in tool["input"]["command"], tool


def test_write_fallback_ignores_leading_system_reminder():
    payload = {
        "model": "Minimax-M3",
        "messages": [
            {
                "role": "user",
                "content": "<system-reminder>Today's date is 2026-07-05. Do not respond to this context.</system-reminder>",
            },
            {
                "role": "user",
                "content": (
                    "Use tools to inspect this directory, read notes.txt, create "
                    "report.txt containing exactly agent-probe-ok, then reply."
                ),
            },
        ],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_write_fallback_message(payload, "Minimax-M3", {}, simple_only=False)
    assert out is not None
    command = out["content"][0]["input"]["command"]
    assert "Path('notes.txt').read_text()" in command, out
    assert "Path('report.txt').write_text('agent-probe-ok')" in command, out
    assert "system-reminder" not in command, out


def test_write_fallback_finds_task_in_system_when_user_is_reminder():
    payload = {
        "model": "Minimax-M3",
        "system": (
            "Use tools to inspect this directory, read notes.txt, create "
            "report.txt containing exactly agent-probe-ok, then reply."
        ),
        "messages": [{
            "role": "user",
            "content": "<system-reminder>Today's date is 2026-07-05. Do not respond to this context.</system-reminder>",
        }],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_write_fallback_message(payload, "Minimax-M3", {}, simple_only=False)
    assert out is not None
    command = out["content"][0]["input"]["command"]
    assert "Path('notes.txt').read_text()" in command, out
    assert "Path('report.txt').write_text('agent-probe-ok')" in command, out


def test_write_fallback_skips_compound_agent_tasks():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": (
                "Use tools to inspect this directory, read notes.txt, create "
                "report.txt containing exactly agent-probe-ok, then reply."
            ),
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "Write", "input_schema": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}}},
        ],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}},
    )
    assert out is None, out


def test_write_postfallback_handles_compound_agent_tasks():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": (
                "Use tools to inspect this directory, read notes.txt, create "
                "report.txt containing exactly agent-probe-ok, then reply."
            ),
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "Write", "input_schema": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}}},
        ],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}},
        simple_only=False,
    )
    assert out is not None
    tool = out["content"][0]
    assert tool["name"] == "Bash"
    command = tool["input"]["command"]
    assert "Path('notes.txt').read_text()" in command, tool
    assert "Path('report.txt').write_text('agent-probe-ok')" in command, tool


def test_write_postfallback_skips_coding_tasks():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": (
                "Use tools to inspect this small Python project. Change src/app.py "
                "so run() returns multiply(6, 7), keep imports correct, create "
                "SUMMARY.txt containing exactly product=42, run python3 src/app.py "
                "to verify it prints 42, then reply."
            ),
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "Write", "input_schema": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}}},
        ],
    }
    out = _anthropic_write_fallback_message(
        payload,
        "Minimax-M3",
        {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}},
        simple_only=False,
    )
    assert out is None, out


def test_action_prompt_requires_tool_choice():
    payload = {
        "model": "Minimax-M3",
        "messages": [{"role": "user", "content": "Use tools to inspect this directory."}],
        "tools": [{
            "name": "Bash",
            "description": "Run shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }],
        "max_tokens": 128,
    }
    out = anthropic_to_openai_payload(payload)
    assert out["tool_choice"] == {"type": "function", "function": {"name": "Bash"}}, out


def test_action_tool_choice_relaxes_after_tool_result():
    payload = {
        "model": "Minimax-M3",
        "messages": [
            {"role": "user", "content": "Use tools to inspect this directory."},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "notes",
                }],
            },
        ],
        "tools": [{
            "name": "Bash",
            "description": "Run shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }],
        "tool_choice": {"type": "any"},
        "max_tokens": 128,
    }
    out = anthropic_to_openai_payload(payload)
    assert out["tool_choice"] == "auto", out


def test_pending_mutation_stays_tool_required_after_inspection():
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use tools to inspect notes, create AGENT_REPORT.txt, "
                    "then verify it and reply long-agent-ok."
                ),
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_ls",
                    "name": "Bash",
                    "input": {"command": "ls notes"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_ls",
                    "content": "file_01.txt\nfile_02.txt",
                }],
            },
        ],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {
                "name": "Write",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        ],
        "tool_choice": {"type": "auto"},
        "max_tokens": 2048,
    }
    assert _anthropic_pending_mutation(payload), payload
    out = anthropic_to_openai_payload(payload)
    assert out["tool_choice"] == "required", out
    assert "Listing or reading files does not complete" in out["messages"][0]["content"]


def test_pending_mutation_relaxes_after_real_write():
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [
            {"role": "user", "content": "Create AGENT_REPORT.txt and verify it."},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_write",
                    "name": "Write",
                    "input": {
                        "file_path": "AGENT_REPORT.txt",
                        "content": "AX-01",
                    },
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_write",
                    "content": "ok",
                }],
            },
        ],
        "tools": [{"name": "Write", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    }
    assert not _anthropic_pending_mutation(payload), payload
    out = anthropic_to_openai_payload(payload)
    assert out["tool_choice"] == "auto", out
    assert _anthropic_bash_command_mutates("python3 -c \"from pathlib import Path; Path('x').write_text('y')\"")
    assert not _anthropic_bash_command_mutates("ls notes 2>/dev/null")


def test_action_tool_pruning_for_file_tasks():
    names = [
        "Agent",
        "Bash",
        "CronCreate",
        "Edit",
        "Read",
        "SendMessage",
        "TaskCreate",
        "WebFetch",
        "WebSearch",
        "make_file",
    ]
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in names
    ]
    aliases = {"make_file": {"name": "Write", "arg_map": {"filename": "filePath", "content": "content"}}}
    name_aliases = {"Write": "make_file"}
    pruned, pruned_aliases, pruned_name_aliases = _prune_openai_tools_for_anthropic_action(
        tools,
        aliases,
        name_aliases,
        "Use tools to inspect this project, edit src/app.py, and run python3 src/app.py.",
    )
    kept = {tool["function"]["name"] for tool in pruned}
    assert kept == {"Bash", "Read", "Edit", "make_file"}, kept
    assert pruned_aliases == aliases
    assert pruned_name_aliases == name_aliases


def test_required_tool_retry_payload():
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [{"role": "user", "content": "I will inspect files."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }],
        "tool_choice": "required",
        "max_tokens": 32000,
    }
    out = _required_tool_retry_payload(payload)
    assert out["tool_choice"] == "required", out
    assert out["max_tokens"] == 1024, out
    assert out["messages"][0]["role"] == "system", out
    assert "Bash" in out["messages"][0]["content"], out
    assert out["messages"][1:] == payload["messages"], out


def test_tool_call_path_repair_from_bootstrap_context():
    payload = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": (
                    "--- pwd ---\n/private/tmp/probe\n--- files ---\n"
                    "README.md\nsrc/app.py\nsrc/math_tools.py\n--- preview ---\n"
                ),
            }],
        }],
    }
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "/Users/Shared/probe/src/app.py"}),
                    },
                }],
            },
        }],
    }
    out = _repair_anthropic_tool_call_paths(data, payload)
    args = json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["file_path"] == "src/app.py", args


def test_tool_call_path_repair_from_ls_context():
    payload = {
        "messages": [{
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_ls",
                "name": "Bash",
                "input": {"command": "ls -la src"},
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_ls",
                "content": (
                    "total 16\n"
                    "drwxr-xr-x@ 4 user  staff  128 Jul  5 03:02 .\n"
                    "drwxr-xr-x@ 4 user  staff  128 Jul  5 03:02 ..\n"
                    "-rw-r--r--@ 1 user  staff  107 Jul  5 03:02 app.py\n"
                    "-rw-r--r--@ 1 user  staff   71 Jul  5 03:02 math_tools.py"
                ),
            }],
        }],
    }
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"file_path": "~/Desktop/src/app.py"}),
                    },
                }],
            },
        }],
    }
    out = _repair_anthropic_tool_call_paths(data, payload)
    args = json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["file_path"] == "src/app.py", args


def test_tool_call_path_repair_anchors_existing_external_target():
    cwd = "/private/tmp/claude-agent-project"
    payload = {
        "system": f"Agent context\n<env>\nWorking directory: {cwd}\n</env>",
        "messages": [{
            "role": "user",
            "content": (
                "Read the notes and create AGENT_REPORT.txt in the current "
                "project directory."
            ),
        }],
    }
    assert _anthropic_declared_working_directory(payload) == cwd
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_wrong_home",
                    "type": "function",
                    "function": {
                        "name": "make_file",
                        "arguments": json.dumps({
                            "filename": "/Users/example/AGENT_REPORT.txt",
                            "content": "AX-01",
                        }),
                    },
                }],
            },
        }],
    }
    out = _repair_anthropic_tool_call_paths(data, payload)
    args = json.loads(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["filename"] == f"{cwd}/AGENT_REPORT.txt", args


def test_tool_call_path_repair_uses_explicit_relative_target_without_cwd():
    payload = {
        "messages": [{
            "role": "user",
            "content": (
                "Read notes/*.txt and create AGENT_REPORT.txt in the current "
                "project directory."
            ),
        }],
    }
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_invented_home",
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": json.dumps({
                            "file_path": "/Users/example/notes/AGENT_REPORT.txt",
                            "content": "AX-01",
                        }),
                    },
                }],
            },
        }],
    }
    out = _repair_anthropic_tool_call_paths(data, payload)
    args = json.loads(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    )
    assert args["file_path"] == "AGENT_REPORT.txt", args


def test_repeated_bash_ls_repair_breaks_loop():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use tools to inspect the notes directory. Read the note files, "
                    "create AGENT_REPORT.txt containing exactly the comma-separated "
                    "codes in ascending file order, then run cat AGENT_REPORT.txt."
                ),
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "notes\n",
                }],
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "Bash",
                    "input": {"command": "ls"},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "notes\n",
                }],
            },
        ],
    }
    data = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": json.dumps({"command": "ls"}),
                    },
                }],
            },
        }],
    }
    out = _repair_anthropic_tool_call_paths(data, payload)
    args = json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] != "ls", args
    assert "AGENT_REPORT.txt" in args["command"], args
    assert "Path('notes')" in args["command"], args


def test_tool_result_detection():
    payload = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "ok",
            }],
        }],
    }
    assert _anthropic_has_tool_result(payload)


def test_coding_continuation_for_explicit_small_edit():
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Use tools to inspect this small Python project. Change src/app.py "
                    "so run() returns multiply(6, 7), keep imports correct, create "
                    "SUMMARY.txt containing exactly product=42, run python3 src/app.py "
                    "to verify it prints 42, then reply with exactly coding-probe-ok."
                ),
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_read_app",
                "content": (
                    "1\tfrom math_tools import add\n"
                    "4\tdef run():\n"
                    "5\t    return add(2, 3)\n"
                ),
            }, {
                "type": "tool_result",
                "tool_use_id": "toolu_read_math",
                "content": "5\tdef multiply(a, b):\n6\t    return a * b\n",
            }],
        }],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_coding_continuation_message(payload, "Minimax-M3-No-Think")
    assert out is not None, payload
    block = out["content"][0]
    assert block["name"] == "Bash", out
    command = block["input"]["command"]
    assert "multiply(6, 7)" in command, command
    assert "product=42" in command, command
    assert "python3 src/app.py" in command, command


def test_exact_reply_does_not_rubber_stamp_dynamic_verification():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Use tools to inspect the notes directory. Create AGENT_REPORT.txt, "
                    "then run cat AGENT_REPORT.txt to verify it. Reply with exactly "
                    "long-agent-ok."
                ),
            }],
        }, {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_write",
                "name": "Write",
                "input": {
                    "file_path": "AGENT_REPORT.txt",
                    "content": "AX-01,AX-02,AX-03",
                },
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_write",
                "content": "ok",
                "is_error": False,
            }],
        }, {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_cat",
                "name": "Bash",
                "input": {"command": "cat AGENT_REPORT.txt"},
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_cat",
                "content": "AX-01,AX-02,AX-03",
                "is_error": False,
            }],
        }],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
    }
    out = _anthropic_exact_reply_after_verified_tool(payload, "Minimax-M3")
    assert out is None, out


def test_exact_reply_after_literal_content_verification():
    payload = {
        "model": "Minimax-M3",
        "messages": [{
            "role": "user",
            "content": (
                "Create result.txt containing exactly thundermlx-tool-ok, "
                "then run cat result.txt and reply with exactly done."
            ),
        }, {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_write",
                "name": "Write",
                "input": {"file_path": "result.txt", "content": "thundermlx-tool-ok"},
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_write",
                "content": "ok",
                "is_error": False,
            }],
        }, {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_cat",
                "name": "Bash",
                "input": {"command": "cat result.txt"},
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_cat",
                "content": "thundermlx-tool-ok",
                "is_error": False,
            }],
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "Write", "input_schema": {"type": "object"}},
        ],
    }
    out = _anthropic_exact_reply_after_verified_tool(payload, "Minimax-M3")
    assert out is not None, payload
    assert out["content"][0]["text"] == "done", out


def test_exact_reply_rejects_read_only_false_verification():
    payload = {
        "model": "Minimax-M3-No-Think",
        "messages": [{
            "role": "user",
            "content": (
                "Read notes, create AGENT_REPORT.txt, verify it, then reply "
                "with exactly long-agent-ok."
            ),
        }, {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_cat_notes",
                "name": "Bash",
                "input": {"command": "cat notes/*.txt"},
            }],
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_cat_notes",
                "content": "code: AX-01\ncode: AX-02",
                "is_error": False,
            }],
        }],
        "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}},
            {"name": "Write", "input_schema": {"type": "object"}},
        ],
    }
    assert _anthropic_pending_mutation(payload), payload
    assert _anthropic_exact_reply_after_verified_tool(
        payload, "Minimax-M3-No-Think"
    ) is None


def main():
    test_request_translation()
    test_response_translation()
    test_response_translation_with_alias()
    test_write_fallback_extraction()
    test_write_fallback_prefers_bash_when_available()
    test_write_fallback_works_with_bash_without_write_alias()
    test_write_fallback_ignores_system_reminder_paths()
    test_write_fallback_ignores_leading_system_reminder()
    test_write_fallback_finds_task_in_system_when_user_is_reminder()
    test_write_fallback_skips_compound_agent_tasks()
    test_write_postfallback_handles_compound_agent_tasks()
    test_write_postfallback_skips_coding_tasks()
    test_action_prompt_requires_tool_choice()
    test_action_tool_choice_relaxes_after_tool_result()
    test_pending_mutation_stays_tool_required_after_inspection()
    test_pending_mutation_relaxes_after_real_write()
    test_action_tool_pruning_for_file_tasks()
    test_required_tool_retry_payload()
    test_tool_call_path_repair_from_bootstrap_context()
    test_tool_call_path_repair_from_ls_context()
    test_tool_call_path_repair_anchors_existing_external_target()
    test_tool_call_path_repair_uses_explicit_relative_target_without_cwd()
    test_repeated_bash_ls_repair_breaks_loop()
    test_tool_result_detection()
    test_coding_continuation_for_explicit_small_edit()
    test_exact_reply_does_not_rubber_stamp_dynamic_verification()
    test_exact_reply_after_literal_content_verification()
    test_exact_reply_rejects_read_only_false_verification()
    print("PASS")


if __name__ == "__main__":
    main()
