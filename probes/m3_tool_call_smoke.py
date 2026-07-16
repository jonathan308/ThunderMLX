#!/usr/bin/env python3
"""OpenAI-compatible tool-call smoke test for MiniMax-M3.

This verifies actual tool invocation, not just prompts that carry tool schemas:
non-stream and stream requests must both return OpenAI `tool_calls` with
`finish_reason=tool_calls`, and streaming output must not leak raw MiniMax XML
tool markers to clients. It also verifies named-function forcing and
`tool_choice=none`, matching the OpenAI semantics added upstream in mlx-vlm
0.6.5.
"""
import argparse
import json
import time
import urllib.request


BASE = "http://127.0.0.1:8080"


def request_json(method, path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(timeout=5):
    out = request_json("GET", "/health", timeout=timeout)
    # The arbiter exposes M3 health under ``m3.health`` while the direct
    # endpoint returns it at the top level. Normalize both so this probe can
    # validate the exact same request path through ports 8080 and 8010.
    nested = (out.get("m3") or {}).get("health") if isinstance(out, dict) else None
    return nested if isinstance(nested, dict) else out


def wait_idle(before_completed, timeout=90):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = health()
        if (
            not last.get("active_request")
            and int(last.get("requests_completed") or 0) > before_completed
        ):
            return last
        time.sleep(0.2)
    return last or health()


def tool_schema():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]


def messages():
    return [
        {
            "role": "system",
            "content": (
                "You are a tool-calling assistant. When asked for weather, "
                "call get_weather. Do not answer from memory."
            ),
        },
        {"role": "user", "content": "Use the tool to get the weather in Paris."},
    ]


def payload(model, stream, max_tokens):
    return {
        "model": model,
        "messages": messages(),
        "tools": tool_schema(),
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": bool(stream),
    }


def run_nonstream(model, max_tokens, timeout):
    before = int(health().get("requests_completed") or 0)
    started = time.time()
    out = request_json(
        "POST",
        "/v1/chat/completions",
        payload(model, False, max_tokens),
        timeout=timeout,
    )
    final = wait_idle(before)
    choice = (out.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    last = final.get("last_request") or {}
    row = {
        "stream": False,
        "elapsed_s": round(time.time() - started, 3),
        "finish": choice.get("finish_reason"),
        "tool_calls": calls,
        "content": message.get("content"),
        "failed": final.get("requests_failed"),
        "decode_tps": last.get("decode_tps"),
        "ttft": last.get("first_token_s"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    if not calls or choice.get("finish_reason") != "tool_calls":
        raise SystemExit("non-stream tool call missing")
    return row


def run_stream(model, max_tokens, timeout):
    before = int(health().get("requests_completed") or 0)
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload(model, True, max_tokens)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    calls = []
    finish = None
    raw_chunks = []
    reasoning_chunks = []
    reasoning_times = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                break
            raw_chunks.append(item)
            evt = json.loads(item)
            for choice in evt.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                calls.extend(delta.get("tool_calls") or [])
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    reasoning_chunks.append(reasoning)
                    reasoning_times.append(time.time() - started)
    final = wait_idle(before)
    last = final.get("last_request") or {}
    leaked = any(
        marker in chunk
        for chunk in raw_chunks
        for marker in ("<tool_call", "<invoke", "]<]minimax")
    )
    row = {
        "stream": True,
        "elapsed_s": round(time.time() - started, 3),
        "finish": finish,
        "tool_calls": calls,
        "raw_chunks": len(raw_chunks),
        "reasoning_chunks": len(reasoning_chunks),
        "first_reasoning_s": (
            round(reasoning_times[0], 3) if reasoning_times else None
        ),
        "reasoning_span_s": (
            round(reasoning_times[-1] - reasoning_times[0], 3)
            if len(reasoning_times) > 1 else 0.0
        ),
        "raw_marker_leak": leaked,
        "failed": final.get("requests_failed"),
        "decode_tps": last.get("decode_tps"),
        "ttft": last.get("first_token_s"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    if not calls or finish != "tool_calls" or leaked:
        raise SystemExit("stream tool call missing or raw marker leaked")
    if "no-think" not in model.lower() and len(reasoning_chunks) < 2:
        raise SystemExit(
            "thinking tool call did not stream incremental reasoning_content"
        )
    return row


def run_tool_choice_semantics(model, max_tokens, timeout):
    choice_tools = tool_schema() + [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }]
    named = request_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Use the selected function for Paris."}
            ],
            "tools": choice_tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_time"},
            },
            "temperature": 0,
            "max_tokens": max(256, max_tokens),
        },
        timeout=timeout,
    )
    named_choice = (named.get("choices") or [{}])[0]
    named_calls = (named_choice.get("message") or {}).get("tool_calls") or []
    named_names = [
        str((call.get("function") or {}).get("name") or "")
        for call in named_calls
    ]
    if named_choice.get("finish_reason") != "tool_calls" or named_names != ["get_time"]:
        raise SystemExit(f"named tool_choice was not enforced: {named}")

    none = request_json(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with exactly TOOL-NONE-OK."}
            ],
            "tools": tool_schema(),
            "tool_choice": "none",
            "temperature": 0,
            "max_tokens": 64,
        },
        timeout=timeout,
    )
    none_choice = (none.get("choices") or [{}])[0]
    none_message = none_choice.get("message") or {}
    if none_message.get("tool_calls") or "TOOL-NONE-OK" not in str(
        none_message.get("content") or ""
    ):
        raise SystemExit(f"tool_choice=none was not enforced: {none}")
    print(json.dumps({
        "tool_choice": True,
        "named_finish": named_choice.get("finish_reason"),
        "named_tools": named_names,
        "none_finish": none_choice.get("finish_reason"),
        "none_tools": none_message.get("tool_calls"),
    }, sort_keys=True), flush=True)


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default="Minimax-M3-No-Think")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    h = health()
    if h.get("status") != "healthy":
        raise SystemExit(f"endpoint unhealthy: {h}")
    run_nonstream(args.model, args.max_tokens, args.timeout)
    run_stream(args.model, args.max_tokens, args.timeout)
    run_tool_choice_semantics(args.model, args.max_tokens, args.timeout)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
