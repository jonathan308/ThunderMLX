#!/usr/bin/env python3
"""Exercise a real multi-round OpenAI tool loop through the M3 gateway.

The schemas and message cadence match OpenAI-compatible coding clients such as
ZCode: stream a tool call, execute it locally, append the assistant/tool result
messages, and continue until the model returns a final answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a UTF-8 text file from the workspace.",
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
                "name": "Write",
                "description": "Create or replace a UTF-8 text file.",
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
                "name": "Edit",
                "description": "Replace exact text in an existing file.",
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
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command in the workspace.",
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


def merge_tool_delta(
    accum: dict[int, dict[str, Any]],
    delta: dict[str, Any],
) -> None:
    index = int(delta.get("index") or 0)
    call = accum.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        call["id"] = str(delta["id"])
    if delta.get("type"):
        call["type"] = str(delta["type"])
    function = delta.get("function") or {}
    if function.get("name"):
        call["function"]["name"] += str(function["name"])
    if function.get("arguments"):
        call["function"]["arguments"] += str(function["arguments"])


def stream_turn(
    base: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], str | None]:
    request = Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            item = line[5:].strip()
            if item == "[DONE]":
                break
            event = json.loads(item)
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(str(delta["content"]))
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if thought:
                    reasoning.append(str(thought))
                for tool_delta in delta.get("tool_calls") or []:
                    merge_tool_delta(tool_calls, tool_delta)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content) or None,
    }
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return message, finish_reason


def workspace_path(root: Path, raw_path: str) -> Path:
    root = root.resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"tool path escaped workspace: {raw_path!r}")
    return resolved


def execute_tool(root: Path, call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    arguments = json.loads(function.get("arguments") or "{}")
    if name == "Read":
        return workspace_path(root, arguments["file_path"]).read_text()
    if name == "Write":
        path = workspace_path(root, arguments["file_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments["content"]))
        return f"Wrote {path.relative_to(root)}"
    if name == "Edit":
        path = workspace_path(root, arguments["file_path"])
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        current = path.read_text()
        if old not in current:
            raise ValueError(f"Edit target not found in {path.relative_to(root)}")
        if arguments.get("replace_all"):
            updated = current.replace(old, new)
        else:
            updated = current.replace(old, new, 1)
        path.write_text(updated)
        return f"Edited {path.relative_to(root)}"
    if name == "Bash":
        proc = subprocess.run(
            str(arguments["command"]),
            cwd=root,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.stdout[-12000:] or f"exit_code={proc.returncode}"
    raise ValueError(f"unexpected tool name: {name!r}")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix="thundermlx_openai_multitool_")).resolve()
    (root / "README.md").write_text(
        "# Native Tool Probe\n\nUse tools to make and verify the requested file.\n"
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Execute requested workspace actions "
                "with the supplied tools. Do not claim success before tool "
                "results prove it."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use tools now. Read README.md, create gate.txt containing "
                "exactly round-one, read it, edit it so it contains exactly "
                "round-one\\nround-two, read it again, run `cat gate.txt`, "
                "then reply with exactly zcode-native-pass. Do not stop after "
                "planning."
            ),
        },
    ]
    names: list[str] = []
    started = time.monotonic()
    try:
        for turn in range(1, args.max_turns + 1):
            message, finish = stream_turn(
                args.base,
                {
                    "model": args.model,
                    "messages": messages,
                    "tools": tool_schemas(),
                    "tool_choice": "auto",
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "stream": True,
                    "metadata": {"session_id": f"openai-multitool-{root.name}"},
                },
                args.timeout,
            )
            calls = message.get("tool_calls") or []
            print(
                json.dumps(
                    {
                        "turn": turn,
                        "finish": finish,
                        "tools": [
                            (call.get("function") or {}).get("name")
                            for call in calls
                        ],
                        "content": message.get("content"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            messages.append(message)
            if not calls:
                final = str(message.get("content") or "").strip()
                if final != "zcode-native-pass":
                    raise AssertionError(
                        f"agent stopped without the expected final: {final!r}"
                    )
                break
            for call in calls:
                name = str((call.get("function") or {}).get("name") or "")
                names.append(name)
                result = execute_tool(root, call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": result,
                    }
                )
        else:
            raise AssertionError(f"agent exceeded {args.max_turns} turns")

        target = root / "gate.txt"
        if target.read_text() != "round-one\nround-two":
            raise AssertionError(f"wrong final artifact: {target.read_text()!r}")
        if len(names) < 4 or not {"Read", "Write", "Edit"}.issubset(names):
            raise AssertionError(f"insufficient tool coverage: {names}")
        row = {
            "model": args.model,
            "elapsed_s": round(time.monotonic() - started, 3),
            "turns": len([message for message in messages if message.get("role") == "assistant"]),
            "tools": names,
            "workdir": str(root),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        print("PASS", flush=True)
        return row
    except Exception:
        print(f"FAILED WORKDIR {root}", flush=True)
        raise
    finally:
        if not args.keep_workdir:
            shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="Minimax-M3")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
