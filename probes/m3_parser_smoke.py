#!/usr/bin/env python3
"""Smoke-check MiniMax reasoning/content marker parsing."""

import json
import os
import pathlib
import sys
import time

import mlx.core as mx


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
os.environ.setdefault("MLX_M3_TOOL_WRITE_CHUNK_MAX_CHARS", "6000")
os.environ.setdefault("MLX_M3_TOOL_WRITE_CHUNK_TARGET_CHARS", "4096")

from sharded_server import (
    _BATCH_PATH_ACTIVE,
    _FORCE_EOS,
    _add_date_system_context,
    _add_tool_system_hint_if_needed,
    _assistant_content_for_template,
    _anchor_command_working_directory,
    _anchor_command_paths_from_read_history,
    _repair_reversed_edit_arguments_after_failure,
    _bound_large_file_write_arguments,
    _buffered_tool_reasoning,
    _remaining_tool_reasoning,
    _require_alternate_work_tool,
    _date_context_for_session,
    _arm_rank0_semantic_eos,
    _filter_looping_control_tools,
    _file_write_chunk_hint,
    _file_write_payload_chars,
    _incomplete_tool_call_budget_reached,
    _looks_like_tool_compat_fallback_content,
    _model_facing_tool_schemas,
    _remember_assistant_reasoning,
    _tool_request_fallback_content,
    _tool_intent_without_call,
    _tool_retry_messages,
    _tool_retry_no_call_budget,
    _tool_retry_prefix_safety,
    _tool_retry_prefers_no_think,
    _tool_retry_thinking_mode,
    _tool_retry_recovery_hint,
    _tool_write_early_stop_chars,
    _tool_loop_steering_diag,
    _tool_loop_steering_text,
    _usable_tool_turn,
    _parse_tool_calls,
    _prompt_cache_allowed_for_generation,
    _prompt_cache_ssd_backing_state,
    _prompt_cache_ssd_restore_backing_state,
    _prompt_cache_ssd_round_capacity,
    _recover_malformed_xml_tool_calls,
    _sanitize_inbound_tool_call_content,
    _sanitize_inbound_message_content,
    _shell_create_file_payload_info,
    _synthesize_bounded_write_scaffold_text,
    _synthesize_explicit_read_tool_call,
    _synthesize_write_command_tool_call,
    _tool_call_complete_for_stop,
    _tool_call_contains_complete_but_invalid,
    _tool_fragment_looks_degenerate,
    _validate_outgoing_tool_calls,
    split_stream_thinking_delta,
    split_thinking_text,
    TOOL_WRITE_CHUNK_TARGET_CHARS,
)
from model_gateway import _openai_response_has_usable_content
from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache


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


def check_tool_retry_preserves_long_prompt_prefix():
    messages = [
        {"role": "system", "content": "native tool schema"},
        {"role": "user", "content": "create the artifact"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "content": "partial result", "tool_call_id": "x"},
    ]
    retry = _tool_retry_messages(messages, "Emit one complete tool call now.")
    assert retry[:-1] == messages, retry
    assert retry[-1] == {
        "role": "user",
        "content": "Emit one complete tool call now.",
    }, retry[-1]
    assert messages[0]["content"] == "native tool schema"

    original = list(range(40000))
    appended = original + [40000, 40001, 40002]
    safe = _tool_retry_prefix_safety(original, appended)
    assert not safe["reset"], safe
    assert safe["common_prefix_tokens"] == len(original), safe
    assert safe["reuse_ratio"] == 1.0, safe

    collapsed = [*range(128), *range(90000, 129872)]
    unsafe = _tool_retry_prefix_safety(original, collapsed)
    assert unsafe["reset"], unsafe
    assert unsafe["common_prefix_tokens"] == 128, unsafe

    medium_original = list(range(26000))
    medium_collapsed = [*range(127), *range(80000, 105873)]
    medium_unsafe = _tool_retry_prefix_safety(
        medium_original,
        medium_collapsed,
    )
    assert medium_unsafe["reset"], medium_unsafe

    short = _tool_retry_prefix_safety(
        list(range(1024)),
        [*range(8), *range(5000, 6016)],
    )
    assert not short["reset"], short
    assert _tool_retry_thinking_mode("enabled", prefer_no_think=True) == "disabled"
    assert _tool_retry_thinking_mode("disabled", prefer_no_think=True) == "disabled"
    assert _tool_retry_thinking_mode("enabled", prefer_no_think=False) == "enabled"
    assert not _tool_retry_prefers_no_think("enabled", 1, 3)
    assert not _tool_retry_prefers_no_think("enabled", 2, 3)
    assert _tool_retry_prefers_no_think("enabled", 3, 3)
    assert _tool_retry_prefers_no_think("enabled", 3, 3, 16384)
    assert not _tool_retry_prefers_no_think("enabled", 3, 3, 16385)
    assert not _tool_retry_prefers_no_think("enabled", 3, 3, 50126)
    assert not _tool_retry_prefers_no_think("disabled", 3, 3)

    raw_tool_reasoning = (
        "I will inspect the available PDF tools.</mm:think>"
        "]<]minimax[>[<tool_call>"
        "]<]minimax[>[<invoke name=\"terminal\">"
        "]<]minimax[>[<command>which pandoc</command>"
        "]<]minimax[>[</invoke>"
        "]<]minimax[>[</tool_call>"
    )
    safe_reasoning = _buffered_tool_reasoning(
        raw_tool_reasoning,
        FakeMiniMaxToolModule,
        "enabled",
    )
    assert safe_reasoning == "I will inspect the available PDF tools.", safe_reasoning
    assert "tool_call" not in safe_reasoning and "minimax" not in safe_reasoning
    assert _remaining_tool_reasoning("plan then call", "plan ") == "then call"
    assert _remaining_tool_reasoning("retry plan", "original plan") == ""


def check_thinking_action_retry_is_not_clipped_at_focused_budget():
    focused = _tool_retry_no_call_budget(
        "disabled",
        action_tool_task=True,
    )
    thinking_action = _tool_retry_no_call_budget(
        "enabled",
        action_tool_task=True,
    )
    thinking_required = _tool_retry_no_call_budget(
        "enabled",
        require_call=True,
    )
    assert focused == 384, focused
    assert thinking_action == 1536, thinking_action
    assert thinking_required == 1536, thinking_required


def check_file_cleanup_promise_requires_a_tool_call():
    assert _tool_intent_without_call(
        "I see - there is trailing garbage. Let me strip it and verify."
    )
    assert _tool_intent_without_call(
        "All nine diagrams are generated. Now let me build the PDF script:"
    )
    assert not _tool_intent_without_call(
        "I stripped the trailing garbage and verified the file."
    )


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


def check_bare_nested_invoke_with_namespaced_args_recovers():
    """Recover ZCode's missing namespace boundary before ``<invoke>``."""
    tools = [{
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>"
        '<invoke name="Read">'
        f"{ns}<file_path>word_stats.py{ns}</file_path>"
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "Read", call
    assert json.loads(call["function"]["arguments"]) == {
        "file_path": "word_stats.py",
    }, call


def check_named_empty_read_uses_unique_reasoning_path():
    tools = [{
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        "There's already a `test_word_stats.py`. Let me read it first."
        "</mm:think>"
        f"{ns}<tool_call>"
        '<invoke name="Read">'
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == (
        "There's already a `test_word_stats.py`. Let me read it first."
        "</mm:think>"
    ), remaining
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "Read", calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "test_word_stats.py",
    }, calls


def check_bare_edit_name_with_xml_arguments_recovers():
    tools = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    }]
    ns = "]<]minimax[>["
    old = '        self.assertEqual(top["dog"], 1)\n        stray()'
    new = '        self.assertEqual(top["dog"], 1)'
    text = (
        "I'll fix the stray line.</mm:think>"
        f"{ns}<tool_call>\n{ns}Edit\n"
        "<file_path>/tmp/test_word_stats.py</file_path>\n"
        f"<new_string>{new}</new_string>\n"
        f"<old_text>{old}</old_text>\n"
        "<replaceAll>false</replaceAll>\n"
        f"</Edit>\n</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "I'll fix the stray line.</mm:think>", remaining
    assert len(calls) == 1, calls
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert calls[0]["function"]["name"] == "Edit", calls
    assert arguments == {
        "file_path": "/tmp/test_word_stats.py",
        "old_string": old,
        "new_string": new,
        "replace_all": False,
    }, arguments
    assert _tool_call_complete_for_stop(text, FakeMiniMaxToolModule, tools)


def check_hybrid_named_parameter_edit_recovers_without_retry():
    """Recover the exact long ZCode Edit drift without rewriting its HTML."""
    tools = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    }]
    ns = "]<]minimax[>["
    old = ".eyebrow{color:red}\n</style>"
    new = ".eyebrow{color:blue}\n.card{display:grid}\n</style>"
    text = (
        f"{ns}<tool_call>{ns}<invoke name=\"Edit\">"
        f"{ns}<command>edit</command>"
        f'<parameter name="path">/tmp/guide.html{ns}</parameter>'
        f'<parameter name="oldtext">{old}{ns}</parameter>'
        f'<parameter name="newtext">{new}{ns}</content>'
        f"{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, (validated, dropped)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {
        "file_path": "/tmp/guide.html",
        "old_string": old,
        "new_string": new,
    }, args

    # The upstream parser can materialize an absent optional boolean as null.
    # It must remain absent, not invalidate an otherwise complete Edit call.
    null_optional = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "arguments": {
                **args,
                "replace_all": None,
            },
        },
    }]
    validated, dropped = _validate_outgoing_tool_calls(
        null_optional,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, (validated, dropped)
    clean_args = json.loads(validated[0]["function"]["arguments"])
    assert "replace_all" not in clean_args, clean_args


def check_quoted_positional_agent_call_recovers():
    """Recover the exact ZCode thinking drift: ``<invoke agent\">``."""
    tools = [{
        "type": "function",
        "function": {
            "name": "Agent",
            "description": "Run a focused subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        },
    }]
    ns = "]<]minimax[>["
    text = (
        "Let me inspect the project.</mm:think>"
        f"{ns}<tool_call>"
        f'{ns}<invoke agent\">{{"description":"Find files",'
        '"prompt":"Find word_stats.py and its tests.",'
        '"subagent_type":"general-purpose"}'
        f"{ns}</prompt}}"
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "Let me inspect the project.</mm:think>", remaining
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["function"]["name"] == "Agent", call
    arguments = json.loads(call["function"]["arguments"])
    assert arguments == {
        "description": "Find files",
        "prompt": "Find word_stats.py and its tests.",
        "subagent_type": "general-purpose",
    }, arguments


def check_relative_read_path_stays_relative_when_home_cwd_is_misleading():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
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
    ]
    processed_messages = [
        {
            "role": "system",
            "content": (
                "<env>\nCurrent working directory: "
                "/Users/tester\n</env>"
            ),
        },
        {
            "role": "user",
            "content": "Add a --min-length option to word_stats.py.",
        },
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "read-relative",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"file_path": "word_stats.py"}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "read-relative",
            "content": "#!/usr/bin/env python3\n",
        },
    ]
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "Read",
                "arguments": {
                    "file_path": "/Users/tester/word_stats.py",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages[:2],
        raw_output="I'll read word_stats.py first.",
    )
    assert dropped == 0, dropped
    read_arguments = json.loads(validated[0]["function"]["arguments"])
    assert read_arguments["file_path"] == "word_stats.py", read_arguments

    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "Edit",
                "arguments": {
                    "file_path": "/Users/example/word_stats.py",
                    "old_string": "before",
                    "new_string": "after",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
        raw_output="I'll update word_stats.py now.",
    )
    assert dropped == 0, dropped
    arguments = json.loads(validated[0]["function"]["arguments"])
    assert arguments["file_path"] == "word_stats.py", arguments


def check_under_specified_positional_edit_is_rejected():
    tools = [{
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
    ns = "]<]minimax[>["
    text = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Edit">'
        f"{ns}<file_path>word_stats.py{ns}</file_path>"
        f"{ns}<param-1>replacement only{ns}</param-1>"
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    assert _recover_malformed_xml_tool_calls(
        text,
        FakeMiniMaxToolModule,
        tools,
    ) == []
    calls, _ = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert calls == [], calls


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


def check_complete_json_call_survives_missing_outer_close():
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
    ns = "]<]minimax[>["
    command = "mkdir -p /tmp/llm_guide && cd /tmp/llm_guide && pwd"
    text = (
        "I will create the working directory."
        f"{ns}<tool_call> {ns}<invoke ```json "
        + json.dumps({"name": "Bash", "input": {"command": command}})
        + f" ```{ns} "
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "I will create the working directory.", remaining
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "Bash", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": command}, args

    # A balanced object followed by substantive bytes is not atomic and must
    # stay on the bounded retry path.
    unsafe = text + "and then append another argument"
    calls, remaining = _parse_tool_calls(unsafe, FakeMiniMaxToolModule, tools)
    assert calls == [], calls
    assert remaining == "I will create the working directory.", remaining

    write_tools = [{
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
    flat = (
        "I'll write sections_a.py now."
        f'{ns}<invoke name="Write", "file_path": "/tmp/wrong.py">'
        f"{ns}<tool_call>"
        + json.dumps({"file_path": "/tmp/wrong.py", "content": "print('ok')\n"})
    )
    calls, remaining = _parse_tool_calls(
        flat,
        FakeMiniMaxToolModule,
        write_tools,
    )
    assert remaining == "I'll write sections_a.py now.", remaining
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "Write", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"file_path": "/tmp/wrong.py", "content": "print('ok')\n"}, args

    # Unknown flat keys must never become executable arguments.
    unsafe_flat = flat.rsplit("{", 1)[0] + json.dumps({"payload": "no schema"})
    calls, _ = _parse_tool_calls(
        unsafe_flat,
        FakeMiniMaxToolModule,
        write_tools,
    )
    assert calls == [], calls


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
    assert f"below {TOOL_WRITE_CHUNK_TARGET_CHARS} characters" in hint, hint
    assert "hard parser ceiling is 6000" in hint, hint
    assert "small valid working scaffold" in hint, hint
    assert _tool_write_early_stop_chars() == 5120

    model_tools = _model_facing_tool_schemas(tools)
    model_content = model_tools[0]["function"]["parameters"]["properties"]["content"]
    original_content = tools[0]["function"]["parameters"]["properties"]["content"]
    assert model_content["maxLength"] == TOOL_WRITE_CHUNK_TARGET_CHARS
    assert "small scaffold" in model_content["description"], model_content
    assert "maxLength" not in original_content, original_content

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
    assert "`file_path`" in edit_recovery, edit_recovery
    assert "`old_string`" in edit_recovery, edit_recovery
    assert "`new_string`" in edit_recovery, edit_recovery
    assert "one focused replacement" in edit_recovery, edit_recovery

    class WrongTypeEditToolModule(FakeMiniMaxToolModule):
        @staticmethod
        def parse_tool_call(text, tools):
            return {
                "name": "Edit",
                "arguments": json.dumps({
                    "file_path": "word_stats.py",
                    "old_string": ["old text"],
                    "new_string": {"replacement": "new text"},
                }),
            }

    wrong_type_edit = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Edit">'
        f"{ns}</invoke>"
        f"{ns}</tool_call>"
    )
    type_recovery = _tool_retry_recovery_hint(
        wrong_type_edit,
        WrongTypeEditToolModule,
        edit_tools,
    )
    assert "`old_string` as JSON string" in type_recovery, type_recovery
    assert "`new_string` as JSON string" in type_recovery, type_recovery
    assert "arrays, objects, or nested tags" in type_recovery, type_recovery

    malformed_large_edit = (
        "I will make one focused addition."
        f"{ns}<tool_call>"
        f'{ns}<invoke>edit\">{{"filePath":"/tmp/snake.html",'
        '"newText":"'
        + ("x" * 7000)
        + '","old_text":""}'
    )
    assert _file_write_payload_chars(
        malformed_large_edit,
        edit_tools,
    ) > 6000, "malformed edit invocation was not bounded"
    malformed_edit_recovery = _tool_retry_recovery_hint(
        malformed_large_edit,
        FakeMiniMaxToolModule,
        edit_tools,
    )
    assert "exactly one valid `Edit` call" in malformed_edit_recovery, (
        malformed_edit_recovery
    )
    assert "exact existing text" in malformed_edit_recovery, (
        malformed_edit_recovery
    )
    malformed_space_name_edit = (
        f"{ns}<tool_call>{ns}<invoke edit>"
        f"{ns}<filePath>/tmp/snake.html{ns}</filePath>"
        f"{ns}<newString>" + ("x" * 7000)
    )
    assert _file_write_payload_chars(
        malformed_space_name_edit,
        edit_tools,
    ) > 6000, "space-name edit invocation was not bounded"

    oversized_valid_edit = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "arguments": {
                "file_path": "/tmp/snake.html",
                "old_string": "anchor",
                "new_string": "x" * 7000,
            },
        },
    }]
    bounded_edits, dropped_edits = _validate_outgoing_tool_calls(
        oversized_valid_edit,
        edit_tools,
        return_dropped=True,
    )
    assert bounded_edits == [], bounded_edits
    assert dropped_edits == 1, dropped_edits

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
        "content": "x" * 5300,
    }
    bounded, original_chars = _bound_large_file_write_arguments(
        "Write",
        oversized_call,
    )
    assert original_chars == 5300, original_chars
    assert len(bounded["content"]) < 6000, bounded
    assert "THUNDERMLX_CONTINUE" in bounded["content"], bounded
    assert bounded["file_path"] == "/tmp/snake.html", bounded

    incomplete_large = (
        f"{ns}<tool_call>"
        f'{ns}<invoke name="Write">'
        f"{ns}<file_path>/tmp/snake.html{ns}</file_path>"
        f"{ns}<content>"
        + ("x" * 5300)
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

    # Native MiniMax shorthand uses the advertised tool name as the element
    # itself. It must hit the same early size guard as <invoke name="Write">.
    incomplete_native_short = (
        f"{ns}<tool_call>"
        f"{ns}<write>"
        f"{ns}<file_path>/tmp/native-short.py{ns}</file_path>"
        f"{ns}<content>"
        + ("x" * 7000)
    )
    assert _file_write_payload_chars(
        incomplete_native_short,
        tools,
    ) > 6000, "native <write> shorthand was not bounded"
    native_short_scaffold = _synthesize_bounded_write_scaffold_text(
        incomplete_native_short,
        tools,
    )
    calls, remaining = _parse_tool_calls(
        native_short_scaffold,
        FakeMiniMaxToolModule,
        tools,
    )
    assert len(calls) == 1, calls
    assert remaining == "", remaining
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["file_path"] == "/tmp/native-short.py", args
    assert "THUNDERMLX_CONTINUE" in args["content"], args

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
    unavailable_search = (
        f"{ns}<tool_call>"
        f"{ns}<invoke file_search>make_diagrams.py</file_search>"
        f"{ns}</tool_call>"
    )
    search_recovery = _tool_retry_recovery_hint(
        unavailable_search,
        FakeMiniMaxToolModule,
        shell_tools,
    )
    assert "`file_search` is not an advertised tool" in search_recovery, search_recovery
    assert "use `Bash`" in search_recovery, search_recovery
    assert "Do not emit `file_search` again" in search_recovery, search_recovery

    promise_recovery = _tool_retry_recovery_hint(
        "I need to update the missing arguments. Let me fix them:",
        FakeMiniMaxToolModule,
        shell_tools,
    )
    assert "promised an action" in promise_recovery, promise_recovery
    assert "exactly one complete tool call" in promise_recovery, promise_recovery
    assert "`Bash`" in promise_recovery, promise_recovery
    assert "Do not narrate" in promise_recovery, promise_recovery

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

    typo, changes = _anchor_command_working_directory(
        "bash",
        {
            "command": "ls -la",
            "workdir": "/workspace/ThunderMLX/opencode-fixturd",
        },
        tools,
        messages,
    )
    assert typo["workdir"] == root, typo
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

    history_messages = [
        {"role": "user", "content": "Inspect the generated PDF and summarize it."},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_workdir_ok",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({
                        "command": "pwd",
                        "workdir": "/private/tmp",
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_workdir_ok",
            "content": "/private/tmp\n",
        },
    ]
    history_repaired, changes = _anchor_command_working_directory(
        "bash",
        {
            "command": "python3 validate.py",
            "workdir": "/private/tmp/thinnerx-gate-does-not-exist",
        },
        tools,
        history_messages,
    )
    assert history_repaired["workdir"] == "/private/tmp", history_repaired
    assert len(changes) == 1, changes


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
    assert patched[-1]["role"] == "system", patched
    assert "Tools remain available" in patched[-1]["content"], patched[-1]
    assert "tool-call format" in patched[-1]["content"], patched[-1]


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
        assert patched[-1]["role"] == "system", patched
        hints.append(patched[-1]["content"])
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


def check_repeated_write_loop_requires_a_different_work_tool():
    import sharded_server as server

    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("Write", "Edit", "Bash", "Read")
    ]
    looped = [{"role": "user", "content": "Build the guide in sections."}]
    for index in range(3):
        call_id = f"write-{index}"
        looped.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "arguments": {
                            "file_path": "/tmp/guide.html",
                            "content": (
                                "<!doctype html><main></main>"
                                "<!-- THUNDERMLX_CONTINUE -->"
                            ),
                        },
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "File written successfully",
            },
        ])

    previous = server.TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS
    try:
        server.TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS = {"Write"}
        diag = _tool_loop_steering_diag(looped, tools)
        assert diag and "repeated_command" in diag["reasons"], diag
        filtered, names = _filter_looping_control_tools(tools, diag)
        assert names == ["Write"], names
        remaining = [item["function"]["name"] for item in filtered]
        assert remaining == ["Edit", "Bash", "Read"], remaining
        request = {"tool_choice": "auto"}
        assert _require_alternate_work_tool(request, filtered, names), request
        assert request["tool_choice"] == "required", request
        assert request["_tool_loop_required_alternate"] is True, request
        diag["filtered_tools"] = names
        hint = _tool_loop_steering_text(diag)
        assert "different available tool" in hint, hint
        assert "repeat Write" in hint, hint
    finally:
        server.TOOL_LOOP_FILTER_REPEATED_WORK_TOOLS = previous


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
    assert "identical_command_result" in diag["reasons"], diag
    assert "force_final" in diag["reasons"], diag
    patched = _add_tool_system_hint_if_needed(looped, {}, tools, tool_loop_diag=diag)
    assert patched[0]["role"] == "system", patched
    assert patched[-1]["role"] == "system", patched
    assert "Tool loop breaker" in patched[-1]["content"], patched[-1]
    assert "tools are unavailable" not in patched[-1]["content"], patched[-1]


def check_identical_command_result_loop_forces_final_only_when_unchanged():
    import sharded_server as server

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

    def messages(results):
        history = [{"role": "user", "content": "Build and validate it."}]
        for index, result in enumerate(results):
            call_id = f"call-{index}"
            history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": {"cmd": "python3 build.py"},
                    },
                }],
            })
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
        return history

    previous_after = server.TOOL_LOOP_FORCE_FINAL_AFTER
    previous_commands = server.TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS
    previous_pairs = server.TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS
    try:
        server.TOOL_LOOP_FORCE_FINAL_AFTER = 0
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS = 0
        server.TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS = 4

        stuck = _tool_loop_steering_diag(messages(["no output"] * 4), tools)
        assert stuck["repeated_command_result_count"] == 4, stuck
        assert "identical_command_result" in stuck["reasons"], stuck
        assert "force_final" in stuck["reasons"], stuck

        productive = _tool_loop_steering_diag(
            messages(["failed", "1 passed", "2 passed", "3 passed"]),
            tools,
        )
        assert productive and "repeated_command" in productive["reasons"], productive
        assert productive["repeated_command_result_count"] == 1, productive
        assert "identical_command_result" not in productive["reasons"], productive
        assert "force_final" not in productive["reasons"], productive
    finally:
        server.TOOL_LOOP_FORCE_FINAL_AFTER = previous_after
        server.TOOL_LOOP_FORCE_FINAL_REPEATED_COMMANDS = previous_commands
        server.TOOL_LOOP_FORCE_FINAL_IDENTICAL_COMMAND_RESULTS = previous_pairs


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
    assert patched[-1]["role"] == "system", patched
    assert "stop calling tools for this single turn" in patched[-1]["content"], patched[-1]
    assert "Do not request another tool" in patched[-1]["content"], patched[-1]


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


def check_json_encoded_array_argument_matches_schema():
    tools = [{
        "type": "function",
        "function": {
            "name": "todowrite",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["todos"],
            },
        },
    }]
    todos = [
        {"content": "Generate diagrams", "status": "in_progress", "priority": "high"},
        {"content": "Build PDF", "status": "pending", "priority": "high"},
    ]
    calls = [{
        "type": "function",
        "function": {
            "name": "todowrite",
            "arguments": {"todos": json.dumps(todos)},
        },
    }]
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {"todos": todos}, args


def check_equals_name_todowrite_beats_native_bash_misparse():
    item_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "status": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["content", "status", "priority"],
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "todowrite",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {"type": "array", "items": item_schema},
                    },
                    "required": ["todos"],
                },
            },
        },
    ]
    ns = "]<]minimax[>["
    raw = (
        f"{ns}<tool_call>"
        f'{ns}<invoke="todowrite">'
        f"{ns}<todos>{ns}<item>"
        f"{ns}<content>Build PDF{ns}</content>"
        f"{ns}<status>pending{ns}</status>"
        f"{ns}<priority>high{ns}</priority>"
        f"{ns}</item>{ns}</todos>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(raw, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "todowrite", calls
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {
        "todos": [{
            "content": "Build PDF",
            "status": "pending",
            "priority": "high",
        }],
    }, args

    wrong_native_call = [{
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": {"command": args["todos"]},
        },
    }]
    assert _validate_outgoing_tool_calls(wrong_native_call, tools) == []


def check_image_generations_bypass_text_prompt_cache():
    assert not _prompt_cache_allowed_for_generation(
        "disabled",
        [1, 2, 3],
        "/private/tmp/example.png",
    )


def check_zcode_long_turn_xml_repairs_are_schema_grounded():
    todo_item = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "status": {"type": "string"},
            "activeForm": {"type": "string"},
            "priority": {"type": "string"},
        },
        "required": ["content", "status", "activeForm"],
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {"type": "array", "items": todo_item},
                    },
                    "required": ["todos"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
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
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
    ]
    ns = "]<]minimax[>["
    reversed_items = (
        f"{ns}<tool_call>{ns}<invoke name=\"TodoWrite\">"
        f"{ns}<todos>{ns}</item>"
        f"{ns}<content>Inspect current state{ns}</content>"
        f"{ns}<status>completed{ns}</status>"
        f"{ns}<activeform>Inspecting current state{ns}</activeform>"
        f"{ns}</item>"
        f"{ns}<content>Update report{ns}</content>"
        f"{ns}<status>in_progress{ns}</status>"
        f"{ns}<activeform>Updating report{ns}</activeform>"
        f"{ns}</item>{ns}</todos>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        reversed_items,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["todos"] == [
        {
            "content": "Inspect current state",
            "status": "completed",
            "activeForm": "Inspecting current state",
        },
        {
            "content": "Update report",
            "status": "in_progress",
            "activeForm": "Updating report",
        },
    ], args

    reversed_container = (
        f"{ns}<tool_call>"
        f"{ns}<invoke name TodoWrite (replacing the stale list)>"
        f"{ns}</todos>{ns}<item>"
        f"{ns}<content>Inspect current project state{ns}</content>"
        f"{ns}<status>in_progress{ns}</status>"
        f"{ns}<activeform>Inspecting current project state{ns}</activeform>"
        f"{ns}<priority>high{ns}</priority>"
        f"{ns}</item>{ns}<item>"
        f"{ns}<content>Run full tests{ns}</content>"
        f"{ns}<status>pending{ns}</status>"
        f"{ns}<activeform>Running full tests{ns}</activeform>"
        f"{ns}<priority>high{ns}</priority>"
        f"{ns}</item>{ns}</todos>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        reversed_container,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["todos"] == [
        {
            "content": "Inspect current project state",
            "status": "in_progress",
            "activeForm": "Inspecting current project state",
            "priority": "high",
        },
        {
            "content": "Run full tests",
            "status": "pending",
            "activeForm": "Running full tests",
            "priority": "high",
        },
    ], args

    long_todo_items = []
    for index in range(8):
        active_form = (
            ""
            if index == 7
            else f"{ns}<activeform>Working item {index}{ns}</activeform>"
        )
        long_todo_items.append(
            f"{ns}<item>"
            f"{ns}<status>{'in_progress' if index == 2 else 'pending'}"
            f"{ns}</status>"
            f"{ns}<content>Work item {index}{ns}</content>"
            f"{active_form}"
            f"{ns}<priority>medium{ns}</priority>"
            f"{ns}</item>"
        )
    long_open_todo = (
        f"{ns}<tool_call>{ns}<invoke name=\"TodoWrite\">"
        f"{ns}<todos>{''.join(long_todo_items)}"
    )
    assert long_open_todo.count(ns) > 32
    assert not _tool_fragment_looks_degenerate(
        long_open_todo,
        FakeMiniMaxToolModule,
    )
    long_complete_todo = (
        f"{long_open_todo}{ns}</todos>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        long_complete_todo,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert len(args["todos"]) == 8, args
    assert args["todos"][-1]["activeForm"] == "Work item 7", args

    marker_spam = f"{ns}<tool_call>" + (ns * 40)
    assert _tool_fragment_looks_degenerate(
        marker_spam,
        FakeMiniMaxToolModule,
    )

    mixed_tools = [
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
    truncated_todo = (
        f"{ns}<tool_call>{ns}<invoke todo>Replace with updated todos"
        f"{ns}</content>{ns}<priority>high{ns}</priority>"
        f"{ns}</item>{ns}</todos>{ns}</invoke>{ns}</tool_call>"
    )
    calls, _ = _parse_tool_calls(
        truncated_todo,
        FakeMiniMaxToolModule,
        mixed_tools,
    )
    assert all(
        call["function"]["name"] != "Bash"
        for call in calls
    ), calls

    labeled_json = (
        f"{ns}<tool_call>{ns}<invoke TodoWrite}}>{ns} "
        'todos: [{"content":"Update report","status":"in_progress",'
        '"activeForm":"Updating report","priority":"high"}]]'
        f"{ns}}} }}{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        labeled_json,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["todos"][0]["content"] == "Update report", args
    assert args["todos"][0]["priority"] == "high", args

    broken_read = (
        f"{ns}<tool_call>{ns}<invoke name=\"Read -f "
        f"/private/tmp/native-tool-gate/REPORT.md>"
        f"{ns}</file_path>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        broken_read,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args == {
        "file_path": "/private/tmp/native-tool-gate/REPORT.md",
    }, args

    malformed_edit = (
        f"{ns}<tool_call>{ns}<invoke name=\"Edit\">"
        f"{ns}<file_path>/private/tmp/native-tool-gate/stats.py"
        f"{ns}</file_path>"
        f"{ns}<old_string>before{ns}</old_string>"
        f"{ns}<new_value>after{ns}</old_string>"
        f"{ns}<new_string>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(
        malformed_edit,
        FakeMiniMaxToolModule,
        tools,
    )
    assert remaining == "", remaining
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["new_string"] == "after", args

    typed_call = [{
        "type": "function",
        "function": {
            "name": "Edit",
            "arguments": {
                "file_path": "/private/tmp/native-tool-gate/stats.py",
                "old_string": "before",
                "new_string": "after",
                "replace_all": "false",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(typed_call, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["replace_all"] is False, args


def check_function_syntax_invoke_recovers_read():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"],
                },
            },
        },
    ]
    ns = "]<]minimax[>["
    path = "/private/tmp/project/build/make_diagrams.py"
    raw = (
        "Let me read the existing file properly.</mm:think>"
        f"{ns}<tool_call>"
        f'{ns}<invoke name="read_file(path="{path}")]'
        f"{ns}</command>{ns}</invoke>{ns}</tool_call>"
    )
    calls, remaining = _parse_tool_calls(raw, FakeMiniMaxToolModule, tools)
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "read", calls
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"filePath": path}, args
    assert "read_file" not in remaining, remaining
    body_raw = (
        f"{ns}<tool_call>"
        f'{ns}<invoke>read_file(path="{path}")]'
        f"{ns}</command>{ns}</invoke>{ns}</tool_call>"
    )
    body_calls, _ = _parse_tool_calls(
        body_raw,
        FakeMiniMaxToolModule,
        tools,
    )
    assert len(body_calls) == 1, body_calls
    assert body_calls[0]["function"]["name"] == "read", body_calls
    assert json.loads(body_calls[0]["function"]["arguments"]) == {
        "filePath": path,
    }

    processed_messages = [
        {
            "role": "system",
            "content": (
                "<env>\nCurrent working directory: /private/tmp/project\n"
                "</env>"
            ),
        },
        {
            "role": "user",
            "content": "Read build/make_diagrams.py, then continue the PDF.",
        },
    ]
    synthesized = _synthesize_explicit_read_tool_call(
        processed_messages,
        tools,
    )
    assert synthesized["function"]["name"] == "read", synthesized
    synthesized_args = json.loads(synthesized["function"]["arguments"])
    assert synthesized_args == {"filePath": "build/make_diagrams.py"}, synthesized_args
    hint = _tool_retry_recovery_hint(
        "Let me read the existing file with the correct path.</mm:think>"
        f"{ns}<tool_call>{ns}<invoke>",
        FakeMiniMaxToolModule,
        tools,
        processed_messages,
    )
    assert "`read`" in hint and "build/make_diagrams.py" in hint, hint

    completed_messages = processed_messages + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "read-1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({
                        "filePath": "/private/tmp/project/build/make_diagrams.py",
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "read-1",
            "content": "def make_diagrams():\n    pass\n",
        },
    ]
    assert _synthesize_explicit_read_tool_call(
        completed_messages,
        tools,
    ) is None
    wrong_command = {
        "command": (
            "cat >> /private/tmp/thundermlx-M3-project/"
            "build/make_diagrams.py <<'EOF'\n# next section\nEOF"
        ),
    }
    anchored, changes = _anchor_command_paths_from_read_history(
        "bash",
        wrong_command,
        tools,
        completed_messages,
    )
    assert changes, (anchored, changes)
    assert "/private/tmp/project/build/make_diagrams.py" in anchored["command"]
    assert "thundermlx-M3-project" not in anchored["command"]


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


def check_missing_write_path_uses_named_tool_proven_target_only():
    tools = [{
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    }]
    root = "/private/tmp/native-tool-gate"
    processed_messages = [
        {
            "role": "system",
            "content": (
                "<env>\nCurrent working directory: " + root + "\n</env>"
            ),
        },
        {
            "role": "user",
            "content": "Fill sections_a.py, sections_b.py, and sections_c.py.",
        },
    ]
    for suffix in ("sections_a.py", "sections_b.py", "sections_c.py"):
        call_id = "read-" + suffix
        processed_messages.extend([
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({
                            "filePath": f"{root}/{suffix}",
                        }),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": "<content>scaffold</content>",
            },
        ])
    call = [{
        "type": "function",
        "function": {
            "name": "write",
            "arguments": {"content": "def add_title():\n    pass\n"},
        },
    }]
    raw = (
        "I'll create a small working scaffold for `sections_a.py` first."
        "]<]minimax[>[<tool_call>"
    )
    validated, dropped = _validate_outgoing_tool_calls(
        call,
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
        raw_output=raw,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["filePath"] == f"{root}/sections_a.py", args
    assert args["content"] == "def add_title():\n    pass\n", args

    # A complete native call can omit all pre-tool prose and drift the parent
    # directory. The one successful same-basename Read remains exact evidence.
    no_cwd_messages = processed_messages[1:]
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": "/private/tmp/thundermlr/sections_a.py",
                    "content": "SECTIONS = []\n",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=no_cwd_messages,
        raw_output="]<]minimax[>[<tool_call>",
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["filePath"] == f"{root}/sections_a.py", args

    leaked_content = (
        "SECTIONS = []\n"
        "]<]minimax[>[\ncontent>"
    )
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": f"{root}/sections_a.py",
                    "content": leaked_content,
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["content"] == "SECTIONS = []", args

    nested_leak = (
        "SECTIONS = []\n"
        "]<]minimax[>[</command>]<]minimax[>[</invoke"
    )
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": f"{root}/sections_a.py",
                    "content": nested_leak,
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["content"] == "SECTIONS = []", args

    mutation_messages = processed_messages[:2] + [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "write-sections-b",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({
                        "filePath": f"{root}/build/sections_b.py",
                        "content": "# first bounded section\n",
                    }),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "write-sections-b",
            "content": "Wrote file successfully.",
        },
    ]
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": f"{root}/sections_b.py",
                    "content": "# second bounded section\n",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=mutation_messages,
        raw_output=(
            "I'll continue `sections_b.py` with the next bounded section."
            "]<]minimax[>[<tool_call>"
        ),
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["filePath"] == f"{root}/build/sections_b.py", args

    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": "/Users/example/Downloads/build_sections_a.py",
                    "content": "def add_title():\n    pass\n",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
        raw_output=raw,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["filePath"] == f"{root}/sections_a.py", args

    ambiguous = raw.replace("sections_a.py", "sections_a.py and sections_b.py")
    validated, dropped = _validate_outgoing_tool_calls(
        call,
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
        raw_output=ambiguous,
    )
    assert validated == [], validated
    assert dropped == 1, dropped

    tagged_path = f"{root}/sections_a.py]<]minimax[>[</filePath>"
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "write",
                "arguments": {
                    "filePath": tagged_path,
                    "content": "def add_title():\n    pass\n",
                },
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["filePath"] == f"{root}/sections_a.py", args


def check_todowrite_fills_only_schema_required_neutral_defaults():
    tools = [{
        "type": "function",
        "function": {
            "name": "todowrite",
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
                                    "enum": ["pending", "in_progress", "completed"],
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
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "todowrite",
                "arguments": {
                    "todos": [{
                        "content": "Implement section A",
                        "status": "in_progress",
                    }],
                },
            },
        }],
        tools,
        return_dropped=True,
    )
    assert dropped == 0, dropped
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["todos"] == [{
        "content": "Implement section A",
        "status": "in_progress",
        "priority": "medium",
    }], args

    # Never invent the task itself when required semantic content is absent.
    validated, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "todowrite",
                "arguments": {"todos": [{"status": "pending"}]},
            },
        }],
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


def check_reversed_write_path_and_content_are_repaired():
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
    payload = (
        "# aggregated content module\n"
        "from pathlib import Path\n\n"
        "def build():\n"
        "    return Path('/tmp/output')\n"
    ) * 4
    calls = [{
        "type": "function",
        "function": {
            "name": "Write",
            "arguments": {
                "file_path": payload,
                "content": "/private/tmp/zcode/build/content_core.py",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(calls, tools)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["file_path"] == "/private/tmp/zcode/build/content_core.py", args
    assert args["content"] == payload.strip(), args

    normal = [{
        "type": "function",
        "function": {
            "name": "Write",
            "arguments": {
                "file_path": "/private/tmp/zcode/build/normal.py",
                "content": "print('ok')\n",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(normal, tools)
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["file_path"].endswith("normal.py"), args
    assert args["content"] == "print('ok')\n", args


def check_multiline_python_c_bash_call_is_executable():
    tools = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["command", "description"],
            },
        },
    }]
    text = (
        "]<]minimax[>[<tool_call>\n"
        "]<]minimax[>[<invoke name=\"Bash\">"
        "]<]minimax[>[<command>cd /private/tmp/zcode/build && python3 -c \"\n"
        "import content_pipeline as cp\n"
        "import content_msa as cm\n"
        "print(cp.PIPELINE_SECTION['title'])\n"
        "print(cm.MSA_DEEP_DIVE['title'])\n"
        "\" 2>&1]<]minimax[>[</command>"
        "]<]minimax[>[<description>Inspect content module shapes</description>"
        "]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    calls, remaining = _parse_tool_calls(text, FakeMiniMaxToolModule, tools)
    assert remaining == "", remaining
    validated, dropped = _validate_outgoing_tool_calls(
        calls,
        tools,
        return_dropped=True,
    )
    assert dropped == 0, (calls, validated, dropped)
    assert len(validated) == 1, validated
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["command"].startswith("cd /private/tmp/zcode/build"), args
    assert "python3 -c" in args["command"], args
    assert args["description"] == "Inspect content module shapes", args


def check_zcode_user_workspace_anchors_drifted_write_and_cd():
    root = "/private/tmp/thundermlx-zcode-artifacts-20260712-2335"
    wrong_root = "/private/tmp/thundermlx-zcode-artifacts-20260712"
    messages = [
        {"role": "system", "content": "Use tools to complete the task."},
        {
            "role": "user",
            "content": f"Work only in the existing {root} directory.",
        },
        {
            "role": "user",
            "content": "Write build/content_manifest.py, then compile it.",
        },
    ]
    write_tools = [{
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
    write_call = [{
        "type": "function",
        "function": {
            "name": "Write",
            "arguments": {
                "file_path": f"{wrong_root}/build/content_manifest.py",
                "content": "TITLE = 'Guide'\n",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(
        write_call,
        write_tools,
        processed_messages=messages,
    )
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["file_path"] == f"{root}/build/content_manifest.py", args

    bash_tools = [{
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
    }]
    bash_call = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "arguments": {
                "command": (
                    f"cd {wrong_root}/build && "
                    "python3 -c \"import content_manifest\""
                ),
                "description": "Validate manifest",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(
        bash_call,
        bash_tools,
        processed_messages=messages,
    )
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["command"].startswith(f"cd {root}/build &&"), args

    list_call = [{
        "type": "function",
        "function": {
            "name": "Bash",
            "arguments": {
                "command": f"ls {wrong_root}/build/ 2>&1",
                "description": "Inspect build directory",
            },
        },
    }]
    validated = _validate_outgoing_tool_calls(
        list_call,
        bash_tools,
        processed_messages=messages,
    )
    args = json.loads(validated[0]["function"]["arguments"])
    assert args["command"] == f"ls {root}/build/ 2>&1", args


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


def check_empty_native_invoke_targets_planned_edit_retry():
    tools = [{
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
    ns = "]<]minimax[>["
    raw = (
        "Good, the edit worked. Let me continue using focused Edit calls."
        "</mm:think>"
        f"{ns}<tool_call>{ns}\ninvoke:{ns}</invoke>{ns}</tool_call>"
    )
    calls, _ = _parse_tool_calls(raw, FakeMiniMaxToolModule, tools)
    assert calls == [], calls
    assert not _usable_tool_turn(
        raw,
        FakeMiniMaxToolModule,
        tools,
        [{"role": "user", "content": "Continue building the guide."}],
        "enabled",
    )
    hint = _tool_retry_recovery_hint(
        raw,
        FakeMiniMaxToolModule,
        tools,
        [{"role": "user", "content": "Continue building the guide."}],
    )
    assert "`Edit`" in hint, hint
    assert "`file_path`" in hint, hint
    assert "`old_string`" in hint, hint
    assert "`new_string`" in hint, hint
    assert "one focused replacement" in hint, hint


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


def check_repeated_long_goal_prompt_with_tool_progress_is_allowed():
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
    repeated = "Inspect this project and continue the goal with tools. " * 120
    messages = []
    for index in range(3):
        call_id = f"call-{index}"
        messages.extend([
            {"role": "user", "content": repeated},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": f"printf {index}"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(index),
            },
        ])
    diag = _tool_loop_steering_diag(messages, tools)
    assert not diag or "repeated_user_tool_prompt" not in diag["reasons"], diag


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


def check_bash_rejects_source_and_numeric_fragments():
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }]
    for command in (
        "20",
        "def build_pdf(path):\n    return path\n",
        "from reportlab.pdfgen import canvas\ncanvas.Canvas('x.pdf')\n",
        '\"\"\"PDF builder.\"\"\"\nimport os\ndef build_pdf(path):\n    return path\n',
        "cat > /tmp/docx_builder.py <<'PYEOF'\nprint('x')\n",
        (
            'Edit</name> <parameter name="file_path">word_stats.py'
            '</parameter> <parameter name="old_string">old</parameter>'
        ),
    ):
        validated, dropped = _validate_outgoing_tool_calls(
            [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": {"command": command},
                },
            }],
            tools,
            return_dropped=True,
        )
        assert validated == [], (command, validated)
        assert dropped == 1, (command, dropped)

    for command in (
        "python3 -c 'print(20)'",
        "python3 - <<'PY'\nprint(20)\nPY",
        "sed -n '1,20p' build_pdf.py",
    ):
        validated, dropped = _validate_outgoing_tool_calls(
            [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": {"command": command},
                },
            }],
            tools,
            return_dropped=True,
        )
        assert dropped == 0, (command, dropped)
        assert len(validated) == 1, (command, validated)

    script = "/tmp/build_pdf.py"
    processed_messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call-python-permission",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": script}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-python-permission",
            "content": f"zsh: permission denied: {script}",
        },
    ]
    repaired, dropped = _validate_outgoing_tool_calls(
        [{
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": {"command": script},
            },
        }],
        tools,
        return_dropped=True,
        processed_messages=processed_messages,
    )
    assert dropped == 0, dropped
    repaired_args = json.loads(repaired[0]["function"]["arguments"])
    assert repaired_args["command"] == f"python3 {script}", repaired_args


def check_ssd_restore_append_capacity_is_bounded():
    logical = 45_178
    old_full_request_reserve = _prompt_cache_ssd_round_capacity(
        logical, 32_768, 256
    )
    bounded_worker_reserve = _prompt_cache_ssd_round_capacity(
        logical, 4_096, 256
    )
    logical_only_artifact = _prompt_cache_ssd_round_capacity(logical, 0, 256)

    assert old_full_request_reserve == 78_080, old_full_request_reserve
    assert bounded_worker_reserve == 49_408, bounded_worker_reserve
    assert logical_only_artifact == 45_312, logical_only_artifact
    assert bounded_worker_reserve < old_full_request_reserve
    assert bounded_worker_reserve - logical <= 4_096 + 255

    cache = MiniMaxM3KVCache()
    cache.kv_cache.keys = mx.zeros(
        (1, 1, old_full_request_reserve, 2), dtype=mx.float16
    )
    cache.kv_cache.values = mx.zeros(
        (1, 1, old_full_request_reserve, 2), dtype=mx.float16
    )
    cache.kv_cache.offset = logical
    cache.index_keys = mx.zeros(
        (1, 1, old_full_request_reserve, 2), dtype=mx.float16
    )
    cache.index_offset = logical

    state, storage = _prompt_cache_ssd_backing_state(
        cache, max_spare_tokens=0
    )
    assert state[0][0].shape[2] == logical_only_artifact
    assert state[1].shape[2] == logical_only_artifact

    restored = MiniMaxM3KVCache()
    restored_storage = _prompt_cache_ssd_restore_backing_state(
        restored,
        state,
        storage,
        target_capacity=logical + 4_096,
    )
    assert restored.offset == logical
    assert restored.index_offset == logical
    assert restored_storage["capacity"] == bounded_worker_reserve
    assert restored.kv_cache.keys.shape[2] == bounded_worker_reserve
    assert restored.index_keys.shape[2] == bounded_worker_reserve
    mx.eval(
        restored.kv_cache.keys,
        restored.kv_cache.values,
        restored.index_keys,
    )


def main():
    check_complete_analysis_channel()
    check_tool_retry_preserves_long_prompt_prefix()
    check_thinking_action_retry_is_not_clipped_at_focused_budget()
    check_file_cleanup_promise_requires_a_tool_call()
    check_stream_analysis_channel()
    check_unknown_channel_does_not_buffer_forever()
    check_image_generations_bypass_text_prompt_cache()
    check_malformed_positional_xml_tool_call_recovers()
    check_bare_nested_invoke_with_namespaced_args_recovers()
    check_named_empty_read_uses_unique_reasoning_path()
    check_bare_edit_name_with_xml_arguments_recovers()
    check_hybrid_named_parameter_edit_recovers_without_retry()
    check_quoted_positional_agent_call_recovers()
    check_relative_read_path_stays_relative_when_home_cwd_is_misleading()
    check_under_specified_positional_edit_is_rejected()
    check_malformed_command_tag_tool_calls_recover()
    check_codex_pseudo_goal_call_recovers()
    check_codex_pseudo_goal_before_malformed_exec_prefers_goal()
    check_invoke_name_attr_drift_recovers()
    check_display_style_tool_call_recovers_and_strips()
    check_loose_segment_command_tool_call_recovers()
    check_incomplete_loose_segment_command_is_not_emitted()
    check_incomplete_native_write_is_never_emitted()
    check_complete_json_call_survives_missing_outer_close()
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
    check_repeated_write_loop_requires_a_different_work_tool()
    check_repeated_exec_command_eventually_forces_final_answer()
    check_identical_command_result_loop_forces_final_only_when_unchanged()
    check_repeated_apply_patch_keeps_tool_schema_stable()
    check_long_tool_loop_gets_force_final_hint()
    check_create_goal_objective_aliases()
    check_update_plan_scalar_coerces_to_plan_item()
    check_json_encoded_array_argument_matches_schema()
    check_equals_name_todowrite_beats_native_bash_misparse()
    check_zcode_long_turn_xml_repairs_are_schema_grounded()
    check_function_syntax_invoke_recovers_read()
    check_missing_required_tool_args_report_dropped()
    check_missing_write_path_uses_named_tool_proven_target_only()
    check_todowrite_fills_only_schema_required_neutral_defaults()
    check_empty_required_tool_args_report_dropped()
    check_parameterless_and_optional_only_tool_args_are_accepted()
    check_malformed_apply_patch_payload_report_dropped()
    check_reversed_write_path_and_content_are_repaired()
    check_multiline_python_c_bash_call_is_executable()
    check_zcode_user_workspace_anchors_drifted_write_and_cd()
    check_tool_call_reasoning_recall_restores_model_context()
    check_read_file_coerces_to_exec_command()
    check_exec_stdin_aliases_to_write_stdin()
    check_malformed_apply_patch_add_file_coerces_to_exec_write()
    check_stale_unavailable_tool_gets_compatibility_fallback()
    check_zcoder_codex_tool_argument_matrix()
    check_empty_tool_markers_get_specific_fallback()
    check_empty_native_invoke_targets_planned_edit_retry()
    check_tool_fallback_content_does_not_poison_next_turn()
    check_gateway_treats_tool_fallback_as_unusable()
    check_repeated_long_user_tool_prompt_forces_final()
    check_repeated_long_goal_prompt_with_tool_progress_is_allowed()
    check_inbound_tool_call_content_is_not_model_facing()
    check_tool_stop_on_valid_or_complete_invalid_tool_call()
    check_invalid_apply_patch_stops_decode_without_emitting_tool()
    check_malformed_apply_patch_simple_write_synthesizes_exec()
    check_semantic_decode_stop_is_rank0_owned()
    check_bash_rejects_source_and_numeric_fragments()
    check_ssd_restore_append_capacity_is_bounded()
    print("PASS")


if __name__ == "__main__":
    main()
