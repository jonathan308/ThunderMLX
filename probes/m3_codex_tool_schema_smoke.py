#!/usr/bin/env python3
"""Live smoke for Codex-style OpenAI tool schemas beyond shell execution."""

from __future__ import annotations

import argparse
import json
import urllib.request


DEFAULT_BASE = "http://127.0.0.1:8080"


def post_json(base: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "create_goal",
                "description": "Create a persistent agent goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "token_budget": {"type": "integer"},
                    },
                    "required": ["objective"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_plan",
                "description": "Update task plan items.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                    },
                                },
                                "required": ["step", "status"],
                            },
                        }
                    },
                    "required": ["plan"],
                },
            },
        },
    ]


def run_case(base: str, model: str, prompt: str, expected: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict structured tool-calling assistant. "
                    "When the user asks for a goal or plan action, call exactly "
                    "one matching tool with valid JSON arguments."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "tools": tools(),
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    out = post_json(base, payload, timeout)
    choice = (out.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or not calls:
        raise AssertionError(f"missing tool call for {expected}: {json.dumps(out)[:4000]}")
    call = calls[0]
    function = call.get("function") or {}
    name = function.get("name")
    if name != expected:
        raise AssertionError(f"expected {expected}, got {name}: {json.dumps(out)[:4000]}")
    args = json.loads(function.get("arguments") or "{}")
    if expected == "create_goal":
        objective = str(args.get("objective") or "")
        if "compatibility" not in objective.lower():
            raise AssertionError(f"bad create_goal args: {args}")
    if expected == "update_plan":
        plan = args.get("plan")
        if not isinstance(plan, list) or not plan or not plan[0].get("step"):
            raise AssertionError(f"bad update_plan args: {args}")
    return {"model": model, "tool": name, "args": args}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    models = args.model or ["Minimax-M3-No-Think", "Minimax-M3"]
    for model in models:
        print(run_case(
            args.base,
            model,
            "Create a goal with objective exactly: Fix compatibility for agent tool calls.",
            "create_goal",
            args.timeout,
        ))
        print(run_case(
            args.base,
            model,
            "Update the plan with one in_progress step named Audit tool calls.",
            "update_plan",
            args.timeout,
        ))
    print("PASS")


if __name__ == "__main__":
    main()
