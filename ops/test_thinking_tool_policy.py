#!/usr/bin/env python3
"""Offline regression gate for thinking-mode tool policy."""

import ast
import json
import logging
import os
import re
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "sharded_server.py"
SOURCE = SRC.read_text()


def load_function(name, namespace):
    tree = ast.parse(SOURCE, filename=str(SRC))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SRC), "exec"), namespace)


def resolve_thinking_mode(request):
    return request.get("thinking_mode", "enabled")


def loop_steering_text(diag):
    return "DYNAMIC LOOP STEER" if diag and diag.get("triggered") else ""


def check(condition, label):
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    return not condition


def main():
    failures = 0
    ns = {
        "ALLOW_THINKING_BUDGET": False,
        "DEFAULT_FREQUENCY_PENALTY": 0,
        "DEFAULT_MIN_P": 0.0,
        "DEFAULT_PRESENCE_PENALTY": 0,
        "DEFAULT_REPETITION_PENALTY": 0,
        "DEFAULT_SEED": 1,
        "DEFAULT_TEMPERATURE": 0.2,
        "DEFAULT_TOP_K": 0,
        "DEFAULT_TOP_P": 1.0,
        "GEN_PARAM_KEYS": (
            "temperature", "top_p", "top_k", "min_p", "seed",
            "repetition_penalty", "thinking_budget",
        ),
        "THINKING_DEFAULT_REPETITION_PENALTY": 1.10,
        "THINKING_MIN_TEMPERATURE": 0.5,
        "TOOL_DEFAULT_MIN_P": 0.0,
        "TOOL_DEFAULT_REPETITION_PENALTY": 1.08,
        "TOOL_DEFAULT_SEED": 1,
        "TOOL_DEFAULT_TEMPERATURE": 0.2,
        "TOOL_DEFAULT_TOP_K": 0,
        "TOOL_DEFAULT_TOP_P": 1.0,
        "_resolve_thinking_mode": resolve_thinking_mode,
        "logger": logging.getLogger("thinking-tool-policy"),
    }
    load_function("_request_generation_params", ns)
    params = ns["_request_generation_params"]
    tools = [{"type": "function", "function": {"name": "exec"}}]

    thinking_tools = params({"thinking_mode": "enabled"}, tools)
    failures += check(
        thinking_tools["temperature"] == 0.2,
        "thinking+tools keeps the tool temperature",
    )
    failures += check(
        thinking_tools["repetition_penalty"] == 1.08,
        "thinking+tools keeps the tool repetition penalty",
    )
    thinking_prose = params({"thinking_mode": "enabled"})
    failures += check(
        thinking_prose["temperature"] == 0.5,
        "thinking prose still receives the temperature floor",
    )
    failures += check(
        thinking_prose["repetition_penalty"] == 1.10,
        "thinking prose still receives its anti-loop penalty",
    )
    explicit = params(
        {"thinking_mode": "enabled", "temperature": 0.35}, tools
    )
    failures += check(
        explicit["temperature"] == 0.35,
        "an explicit OpenAI temperature remains authoritative",
    )

    hint_ns = {
        "_TOOL_ACTION_VERBS": (
            "add|audit|build|check|copy|create|delete|edit|execute|explore|"
            "fetch|find|fix|implement|inspect|install|list|make|modify|move|"
            "open|patch|read|remove|rename|review|rewrite|run|save|search|"
            "test|update|validate|verify|write"
        ),
        "TOOL_SYSTEM_HINT_ENABLED": True,
        "TOOL_SYSTEM_HINT_TEXT": "STATIC TOOL PRIMER",
        "json": json,
        "os": os,
        "re": re,
        "_resolve_thinking_mode": resolve_thinking_mode,
        "_tool_choice_disables_tools": lambda request: (
            request.get("tool_choice") == "none"
        ),
        "_tool_loop_steering_text": loop_steering_text,
    }
    load_function("_tool_choice_required_name", hint_ns)
    load_function("_last_user_text", hint_ns)
    load_function("_last_user_instruction_text", hint_ns)
    load_function("_tool_text_requests_action", hint_ns)
    load_function("_tool_request_requires_call", hint_ns)
    load_function("_tool_intent_without_call", hint_ns)
    load_function("_tool_working_directory_from_messages", hint_ns)
    load_function("_add_tool_system_hint_if_needed", hint_ns)
    add_hint = hint_ns["_add_tool_system_hint_if_needed"]
    messages = [
        {
            "role": "system",
            "content": (
                "Agent context\n<env>\n  Working directory: "
                "/Users/tester/project\n  Platform: darwin\n</env>"
            ),
        },
        {"role": "user", "content": "work"},
    ]

    initial_thinking = add_hint(
        messages, {"thinking_mode": "enabled"}, tools
    )
    failures += check(
        "/Users/tester/project" in initial_thinking[0]["content"]
        and "STATIC TOOL PRIMER" not in initial_thinking[0]["content"],
        "thinking tool turns get the exact client path anchor",
    )
    failures += check(
        hint_ns["_tool_working_directory_from_messages"](messages)
        == "/Users/tester/project",
        "the OpenCode-style env working directory is extracted exactly",
    )
    steered_thinking = add_hint(
        [{"role": "user", "content": "work"}],
        {"thinking_mode": "enabled"},
        tools,
        {"triggered": True, "reasons": ["repeated_tool"]},
    )
    failures += check(
        steered_thinking[0]["content"] == "DYNAMIC LOOP STEER",
        "thinking tool loops still receive dynamic recovery steering",
    )
    no_path_thinking = add_hint(
        [{"role": "user", "content": "work"}],
        {"thinking_mode": "enabled"},
        tools,
    )
    failures += check(
        no_path_thinking == [{"role": "user", "content": "work"}],
        "thinking prompts without anchored path metadata stay untouched",
    )
    action_no_path = add_hint(
        [{"role": "user", "content": "Create a complete app.py file."}],
        {"thinking_mode": "enabled"},
        tools,
    )
    failures += check(
        "Tool execution rule" in action_no_path[0]["content"],
        "clear thinking action requests receive a compact execution rule",
    )
    action_after_tool = add_hint(
        [
            {"role": "user", "content": "Create a complete app.py file."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
        ],
        {"thinking_mode": "enabled"},
        tools,
    )
    failures += check(
        action_after_tool[0]["content"] == action_no_path[0]["content"],
        "action-task discipline stays byte-stable after tool results",
    )
    action_after_gateway_tool = add_hint(
        [
            {"role": "user", "content": "Create a complete app.py file."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
            {"role": "user", "content": "Tool result x:\ninspection output"},
        ],
        {"thinking_mode": "enabled"},
        tools,
    )
    failures += check(
        action_after_gateway_tool[0]["content"] == action_no_path[0]["content"],
        "gateway-wrapped tool results keep action discipline byte-stable",
    )
    explanation_no_path = add_hint(
        [{"role": "user", "content": "Explain how to create an app.py file."}],
        {"thinking_mode": "enabled"},
        tools,
    )
    failures += check(
        explanation_no_path
        == [{"role": "user", "content": "Explain how to create an app.py file."}],
        "explanatory prompts do not force a tool call",
    )
    initial_no_think = add_hint(
        messages, {"thinking_mode": "disabled"}, tools
    )
    failures += check(
        initial_no_think[0]["content"].startswith("STATIC TOOL PRIMER\n\n")
        and "/Users/tester/project" in initial_no_think[0]["content"],
        "no-thinking tool turns retain the primer and exact path anchor",
    )
    steered_no_think = add_hint(
        messages,
        {"thinking_mode": "disabled"},
        tools,
        {"triggered": True, "reasons": ["repeated_tool"]},
    )
    failures += check(
        steered_no_think[0]["content"]
        == initial_no_think[0]["content"] + "\n\nDYNAMIC LOOP STEER",
        "no-thinking loops combine primer, path anchor, and steering",
    )
    disabled = add_hint(
        messages,
        {"thinking_mode": "disabled", "tool_choice": "none"},
        tools,
    )
    failures += check(
        disabled == messages,
        "tool_choice=none suppresses all tool hints",
    )

    requires_call = hint_ns["_tool_request_requires_call"]
    failures += check(
        requires_call(
            [{"role": "user", "content": "Build a playable Snake game."}],
            {"tool_choice": "auto"},
        ),
        "imperative first-turn work is classified as requiring execution",
    )
    failures += check(
        requires_call(
            [{"role": "user", "content": '"Build a playable Snake game."'}],
            {"tool_choice": "auto"},
        ),
        "OpenCode CLI quote wrappers do not hide an imperative action",
    )
    failures += check(
        requires_call(
            [{"role": "user", "content": "What is 2 + 2?"}],
            {"tool_choice": "required"},
        ),
        "OpenAI tool_choice=required remains authoritative",
    )
    failures += check(
        not requires_call(
            [
                {"role": "user", "content": "Build a playable Snake game."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "content": "ok"},
            ],
            {"tool_choice": "auto"},
        ),
        "a completed tool round permits the normal final prose answer",
    )
    intent_without_call = hint_ns["_tool_intent_without_call"]
    failures += check(
        intent_without_call(
            "<mm:think>I found duplicates.</mm:think>\n"
            "I noticed the file needs cleanup. Let me rewrite the file cleanly."
        ),
        "post-thinking promises to mutate are classified as missing calls",
    )
    failures += check(
        not intent_without_call(
            "<mm:think>Checked.</mm:think>\n"
            "Done. The file is valid and ready."
        ),
        "normal final answers are not classified as missing calls",
    )

    path_ns = {
        "_FILE_PATH_ARGUMENT_KEYS": (
            "filePath", "file_path", "path", "filename", "file", "target"
        ),
        "_MUTATING_FILE_TOOL_NAMES": {
            "applypatch", "edit", "editfile", "makefile", "multiedit",
            "write", "writefile"
        },
        "_last_user_text": hint_ns["_last_user_text"],
        "_last_user_instruction_text": hint_ns[
            "_last_user_instruction_text"
        ],
        "_RELATIVE_MUTATION_TARGET_RE": re.compile(
            r"\b(?:create|write|save|edit|update|modify|patch|rename|move|copy)\s+"
            r"(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:text\s+)?(?:file\s+)?"
            r"(?:named\s+|called\s+|at\s+|to\s+)?"
            r"(?P<path>(?:[A-Za-z0-9_.-]+/)*"
            r"[A-Za-z0-9_.-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)",
            re.IGNORECASE,
        ),
        "_tool_schema_required_names": lambda _tools, name: (
            {"make_file": ["filename", "content"]}.get(name, [])
        ),
        "_tool_working_directory_from_messages": hint_ns[
            "_tool_working_directory_from_messages"
        ],
        "os": os,
        "re": re,
    }
    load_function("_fill_missing_mutating_tool_path", path_ns)
    load_function("_anchor_mutating_tool_paths", path_ns)
    load_function("_tool_request_path_violation", path_ns)
    anchor_paths = path_ns["_anchor_mutating_tool_paths"]
    path_violation = path_ns["_tool_request_path_violation"]
    fill_missing_path = path_ns["_fill_missing_mutating_tool_path"]
    gateway_messages = [
        messages[0],
        {
            "role": "user",
            "content": "Inspect notes and create AGENT_REPORT.txt containing the codes.",
        },
        {"role": "user", "content": "Tool result toolu_ls:\nfile_01.txt"},
    ]
    repaired_missing, missing_changes = fill_missing_path(
        "make_file",
        {"content": "AX-01"},
        [{"type": "function", "function": {"name": "make_file"}}],
        gateway_messages,
    )
    failures += check(
        repaired_missing["filename"]
        == "/Users/tester/project/AGENT_REPORT.txt"
        and len(missing_changes) == 1,
        "one user-named target safely fills a missing required filename",
    )
    ambiguous_missing, ambiguous_changes = fill_missing_path(
        "make_file",
        {"content": "x"},
        [{"type": "function", "function": {"name": "make_file"}}],
        [
            messages[0],
            {"role": "user", "content": "Create first.txt and create second.txt."},
        ],
    )
    failures += check(
        "filename" not in ambiguous_missing and ambiguous_changes == [],
        "multiple user file targets are never guessed",
    )
    anchored, changes = anchor_paths(
        "write",
        {"filePath": "/Users/remembered/Downloads/app.py"},
        [
            messages[0],
            {"role": "user", "content": "Create app.py in the current project."},
        ],
    )
    failures += check(
        anchored["filePath"] == "/Users/tester/project/app.py"
        and len(changes) == 1,
        "a user-named relative file is anchored into the client project",
    )
    nested, nested_changes = anchor_paths(
        "write",
        {"filePath": "/Users/remembered/project/app.py"},
        [
            messages[0],
            {"role": "user", "content": "Create src/app.py in the project."},
        ],
    )
    failures += check(
        nested["filePath"] == "/Users/tester/project/src/app.py"
        and len(nested_changes) == 1,
        "a user-named nested relative path preserves its project suffix",
    )
    failures += check(
        not path_violation(
            "write", {"filePath": "src/app.py"}, messages
        ),
        "relative mutating paths remain valid",
    )
    failures += check(
        not path_violation(
            "write",
            {"filePath": "/Users/tester/project/src/app.py"},
            messages,
        ),
        "absolute paths inside the client working directory remain valid",
    )
    failures += check(
        bool(
            path_violation(
                "write",
                {"filePath": "/Users/other/Downloads/app.py"},
                messages,
            )
        ),
        "invented external mutation paths are rejected",
    )
    failures += check(
        bool(
            path_violation(
                "write",
                {"filePath": "body { color: red; }\n" * 200},
                messages,
            )
        ),
        "content-shaped or oversized mutating paths are rejected",
    )
    explicit_external = [
        messages[0],
        {
            "role": "user",
            "content": "Write /Users/other/Downloads/app.py exactly.",
        },
    ]
    failures += check(
        not path_violation(
            "write",
            {"filePath": "/Users/other/Downloads/app.py"},
            explicit_external,
        ),
        "user-explicit external mutation paths remain valid",
    )
    unchanged, external_changes = anchor_paths(
        "write",
        {"filePath": "/Users/other/Downloads/app.py"},
        explicit_external,
    )
    failures += check(
        unchanged["filePath"] == "/Users/other/Downloads/app.py"
        and external_changes == [],
        "explicit absolute user targets are never rewritten",
    )

    failures += check(
        SOURCE.count("and n >= runaway_budget") == 2,
        "both generation paths use the scoped runaway budget",
    )
    failures += check(
        SOURCE.count("no-call tool guard at token") == 2,
        "both generation paths use the synchronized no-call guard",
    )
    failures += check(
        SOURCE.count("BATCH_TOOL_NATURAL_DRAIN") == 5,
        "completed batch tool calls drain naturally on both ranks",
    )
    print(f"\n{35 - failures}/35 thinking-tool policy checks green")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
